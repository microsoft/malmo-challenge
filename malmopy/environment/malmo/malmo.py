# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from __future__ import absolute_import

import json
import xml.etree.ElementTree
from MalmoPython import AgentHost, ClientPool, ClientInfo, MissionSpec, MissionRecordSpec
from collections import Sequence
from time import sleep

import six
from PIL import Image
from numpy import zeros, log

from ..environment import VideoCapableEnvironment, StateBuilder


def allocate_remotes(remotes):
    """
    Utility method for building a Malmo ClientPool.
    Using this method allows sharing the same ClientPool across
    mutiple experiment
    :param remotes: tuple or array of tuples. Each tuple can be (), (ip,), (ip, port)
    :return: Malmo ClientPool with all registered clients
    """
    if not isinstance(remotes, list):
        remotes = [remotes]

    pool = ClientPool()
    for remote in remotes:
        if isinstance(remote, ClientInfo):
            pool.add(remote)
        elif isinstance(remote, Sequence):
            if len(remote) == 0:
                pool.add(ClientInfo('localhost', 10000))
            elif len(remote) == 1:
                pool.add(ClientInfo(remote[0], 10000))
            else:
                pool.add(ClientInfo(remote[0], int(remote[1])))
    return pool


class TurnState(object):
    def __init__(self):
        self._turn_key = None
        self._has_played = False

    def update(self, key):
        self._has_played = False
        self._turn_key = key

    @property
    def can_play(self):
        return self._turn_key is not None and not self._has_played

    @property
    def key(self):
        return self._turn_key

    @property
    def has_played(self):
        return self._has_played

    @has_played.setter
    def has_played(self, value):
        self._has_played = bool(value)


class MalmoStateBuilder(StateBuilder):
    """
    Base class for specific state builder inside the Malmo platform.
    The #build method has access to the currently running environment and all
    the properties exposed to this one.
    """

    def __call__(self, *args, **kwargs):
        assert isinstance(args[0], MalmoEnvironment), 'provided argument should inherit from MalmoEnvironment'
        return self.build(*args)


class MalmoRGBStateBuilder(MalmoStateBuilder):
    """
    Generate RGB frame state resizing to the specified width/height and depth
    """

    def __init__(self, width, height, grayscale):
        assert width > 0, 'width should be > 0'
        assert width > 0, 'height should be > 0'

        self._width = width
        self._height = height
        self._gray = bool(grayscale)

    def build(self, environment):
        import numpy as np

        img = environment.frame

        if img is not None:
            img = img.resize((self._width, self._height))

            if self._gray:
                img = img.convert('L')
            return np.array(img)
        else:
            return zeros((self._width, self._height, 1 if self._gray else 3)).squeeze()


class MalmoALEStateBuilder(MalmoRGBStateBuilder):
    """
    Commodity class for generating Atari Learning Environment compatible states.

    Properties:
        - depth: Grayscale image
        - width: 84
        - height: 84

    return (84, 84) numpy array
    """

    def __init__(self):
        super(MalmoALEStateBuilder, self).__init__(84, 84, True)


class MalmoEnvironment(VideoCapableEnvironment):
    """
    Interaction with Minecraft through the Malmo Mod.
    """

    MAX_START_MISSION_RETRY = 50

    def __init__(self, mission, actions, remotes,
                 role=0, exp_name="", turn_based=False,
                 recording_path=None, force_world_reset=False):

        assert isinstance(mission, six.string_types), "mission should be a string"
        super(MalmoEnvironment, self).__init__()

        self._agent = AgentHost()
        self._mission = MissionSpec(mission, True)

        # validate actions
        self._actions = actions
        assert actions is not None, "actions cannot be None"
        assert isinstance(actions, Sequence), "actions should be an iterable object"
        assert len(actions) > 0, "len(actions) should be > 0"

        # set up recording if requested
        if recording_path:
            self._recorder = MissionRecordSpec(recording_path)
            self._recorder.recordCommands()
            self._recorder.recordMP4(12, 400000)
            self._recorder.recordRewards()
            self._recorder.recordObservations()
        else:
            self._recorder = MissionRecordSpec()

        self._clients = allocate_remotes(remotes)

        self._force_world_reset = force_world_reset
        self._role = role
        self._exp_name = exp_name
        self._turn_based = bool(turn_based)
        self._turn = TurnState()

        self._world = None
        self._world_obs = None
        self._previous_action = None
        self._last_frame = None
        self._action_count = None
        self._end_result = None

    @property
    def available_actions(self):
        return len(self._actions)

    @property
    def state(self):
        raise NotImplementedError()

    @property
    def end_result(self):
        return self._end_result

    @property
    def reward(self):
        return 0.

    @property
    def done(self):
        latest_ws = self._agent.peekWorldState()
        return latest_ws.has_mission_begun and not latest_ws.is_mission_running

    @property
    def action_count(self):
        return self._action_count

    @property
    def previous_action(self):
        return self._previous_action

    @property
    def frame(self):
        latest_ws = self._agent.peekWorldState()

        if hasattr(latest_ws, 'video_frames') and len(latest_ws.video_frames) > 0:
            self._last_frame = latest_ws.video_frames[-1]

        return Image.frombytes('RGB',
                               (self._last_frame.width, self._last_frame.height),
                               bytes(self._last_frame.pixels))

    @property
    def recording(self):
        return super(MalmoEnvironment, self).recording

    @recording.setter
    def recording(self, val):
        self._recording = bool(val)

        if self.recording:
            if not self._mission.isVideoRequested(0):
                self._mission.requestVideo(212, 160)

    @property
    def is_turn_based(self):
        return self._turn_based

    @property
    def world_observations(self):
        latest_ws = self._agent.peekWorldState()
        if latest_ws.number_of_observations_since_last_state > 0:
            self._world_obs = json.loads(latest_ws.observations[-1].text)

        return self._world_obs

    def _ready_to_act(self, world_state):
        if not self._turn_based:
            return True
        else:
            if not world_state.is_mission_running:
                return False

            if world_state.number_of_observations_since_last_state > 0:
                data = json.loads(world_state.observations[-1].text)
                turn_key = data.get(u'turn_key', None)

                if turn_key is not None and turn_key != self._turn.key:
                    self._turn.update(turn_key)
            return self._turn.can_play

    def do(self, action_id):
        assert 0 <= action_id <= self.available_actions, \
            "action %d is not valid (should be in [0, %d[)" % (action_id,
                                                               self.available_actions)
        action = self._actions[action_id]
        assert isinstance(action, six.string_types)

        if self._turn_based:
            if self._turn.can_play:
                self._agent.sendCommand(str(action), str(self._turn.key))
                self._turn.has_played = True
                self._previous_action = action
                self._action_count += 1
        else:
            self._agent.sendCommand(action)
            self._previous_action = action
            self._action_count += 1

        self._await_next_obs()
        return self.state, sum([reward.getValue() for reward in self._world.rewards]), self.done

    def reset(self):
        super(MalmoEnvironment, self).reset()

        if self._force_world_reset:
            self._mission.forceWorldReset()

        self._world = None
        self._world_obs = None
        self._previous_action = None
        self._action_count = 0
        self._turn = TurnState()
        self._end_result = None

        # Wait for the server (role = 0) to start
        sleep(.5)

        for i in range(MalmoEnvironment.MAX_START_MISSION_RETRY):
            try:
                self._agent.startMission(self._mission,
                                         self._clients,
                                         self._recorder,
                                         self._role,
                                         self._exp_name)
                break
            except Exception as e:
                if i == MalmoEnvironment.MAX_START_MISSION_RETRY - 1:
                    raise Exception("Unable to connect after %d tries %s" %
                                       (self.MAX_START_MISSION_RETRY, e))
                else:
                    sleep(log(i + 1) + 1)

        # wait for mission to begin
        self._await_next_obs()
        return self.state

    def _await_next_obs(self):
        """
        Ensure that an update to the world state is received
        :return:
        """
        # Wait until we have everything we need
        current_state = self._agent.peekWorldState()
        while not self.is_valid(current_state) or not self._ready_to_act(current_state):

            if current_state.has_mission_begun and not current_state.is_mission_running:
                if not current_state.is_mission_running and len(current_state.mission_control_messages) > 0:
                    # Parse the mission ended message:
                    mission_end_tree = xml.etree.ElementTree.fromstring(current_state.mission_control_messages[-1].text)
                    ns_dict = {"malmo": "http://ProjectMalmo.microsoft.com"}
                    hr_stat = mission_end_tree.find("malmo:HumanReadableStatus", ns_dict).text
                    self._end_result = hr_stat
                break

            # Peek fresh world state from socket
            current_state = self._agent.peekWorldState()

        # Flush current world as soon as we have the entire state
        self._world = self._agent.getWorldState()

        if self._world.is_mission_running:
            new_world = json.loads(self._world.observations[-1].text)
            if new_world is not None:
                self._world_obs = new_world

        # Update video frames if any
        if hasattr(self._world, 'video_frames') and len(self._world.video_frames) > 0:
            self._last_frame = self._world.video_frames[-1]

    def is_valid(self, world_state):
        """
        Check whether the provided world state is valid.
        @override to customize checks
        """

        # observation cannot be empty
        return world_state is not None \
               and world_state.has_mission_begun \
               and len(world_state.observations) > 0 \
               and (len(world_state.rewards) > 0 or self._action_count == 0)

