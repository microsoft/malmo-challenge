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
import re

import numpy as np

from common import Entity, ENV_ACTIONS, ENV_BOARD, ENV_ENTITIES, ENV_BOARD_SHAPE, ENV_AGENT_NAMES

from MalmoPython import MissionSpec
from malmopy.environment.malmo import MalmoEnvironment, MalmoStateBuilder


class PigChaseSymbolicStateBuilder(MalmoStateBuilder):
    """
    This class build a symbolic representation of the current environment.
    Generated states consist of a string array, with the name of the block/entities on the given block.
    """

    def __init__(self, entities_override=True):
        self._entities_override = bool(entities_override)

    def build(self, environment):
        """
        Return a symbolic view of the board

        :param environment Reference to the pig chase environment
        :return (board, entities) where board is an array of shape (9, 9) with the block type / entities name for each coordinate, and entities is a list of current entities
        """
        assert isinstance(environment, PigChaseEnvironment), 'environment is not a Pig Chase Environment instance'

        world_obs = environment.world_observations
        if world_obs is None or ENV_BOARD not in world_obs:
            return None

        # Generate symbolic view
        board = np.array(world_obs[ENV_BOARD], dtype=object).reshape(ENV_BOARD_SHAPE)
        entities = world_obs[ENV_ENTITIES]

        if self._entities_override:
            for entity in entities:
                board[int(entity['z'] + 1), int(entity['x'])] += '/' + entity['name']

        return (board, entities)

class PigChaseEnvironment(MalmoEnvironment):
    """
    Represent the Pig chase with two agents and a pig. Agents can try to catch
    the pig (high reward), or give up by leaving the pig pen (low reward).
    """

    AGENT_TYPE_0 = 0
    AGENT_TYPE_1 = 1
    AGENT_TYPE_2 = 2
    AGENT_TYPE_3 = 3

    VALID_START_POSITIONS = [
        (2.5,1.5), (3.5,1.5), (4.5,1.5), (5.5,1.5), (6.5,1.5),
        (2.5,2.5),            (4.5,2.5),            (6.5,2.5),
        (2.5,3.5), (3.5,3.5), (4.5,3.5), (5.5,3.5), (6.5,3.5),
        (2.5,4.5),            (4.5,4.5),            (6.5,4.5),
        (2.5,5.5), (3.5,5.5), (4.5,5.5), (5.5,5.5), (6.5,5.5)
        ]
    VALID_YAW = [0, 90, 180, 270]

    def __init__(self, remotes,
                 state_builder,
                 actions=ENV_ACTIONS,
                 role=0, exp_name="",
                 human_speed=False, randomize_positions=False):

        assert isinstance(state_builder, MalmoStateBuilder)

        self._mission_xml = open('pig_chase.xml', 'r').read()
        # override tics per ms to play at human speed
        if human_speed:
            print('Setting mission to run at human speed')
            self._mission_xml = re.sub('<MsPerTick>\d+</MsPerTick>',
                                       '<MsPerTick>50</MsPerTick>', self._mission_xml)
        super(PigChaseEnvironment, self).__init__(self._mission_xml, actions,
                                                  remotes, role, exp_name, True)

        self._internal_symbolic_builder = PigChaseSymbolicStateBuilder()
        self._user_defined_builder = state_builder

        self._randomize_positions = randomize_positions
        self._agent_type = None

    @property
    def state(self):
        return self._user_defined_builder.build(self)

    @property
    def done(self):
        """
        Done if we have caught the pig
        """
        return super(PigChaseEnvironment, self).done

    def _construct_mission(self):
        # set agent helmet
        original_helmet = "iron_helmet"
        if self._role == 0:
            original_helmet = "diamond_helmet"
        new_helmet = original_helmet
        if self._agent_type == PigChaseEnvironment.AGENT_TYPE_0:
            new_helmet = "iron_helmet"
        elif self._agent_type == PigChaseEnvironment.AGENT_TYPE_1:
            new_helmet = "golden_helmet"
        elif self._agent_type == PigChaseEnvironment.AGENT_TYPE_2:
            new_helmet = "diamond_helmet"
        elif self._agent_type == PigChaseEnvironment.AGENT_TYPE_3:
            new_helmet = "leather_helmet"
        xml = re.sub(r'type="%s"' % original_helmet,
                     r'type="%s"' % new_helmet, self._mission_xml)
        # set agent starting pos
        if self._randomize_positions and self._role == 0:
            pos = [PigChaseEnvironment.VALID_START_POSITIONS[i]
                    for i in np.random.choice(
                        range(len(PigChaseEnvironment.VALID_START_POSITIONS)),
                                   3, replace=False)]
            while not (self._get_pos_dist(pos[0], pos[1]) > 1.1 and
                       self._get_pos_dist(pos[1], pos[2]) > 1.1 and
                       self._get_pos_dist(pos[0], pos[2]) > 1.1):
                pos = [PigChaseEnvironment.VALID_START_POSITIONS[i]
                        for i in np.random.choice(
                            range(len(PigChaseEnvironment.VALID_START_POSITIONS)),
                                       3, replace=False)]

            xml = re.sub(r'<DrawEntity[^>]+>',
                         r'<DrawEntity x="%g" y="4" z="%g" type="Pig"/>' % pos[0], xml)
            xml = re.sub(r'(<Name>%s</Name>\s*<AgentStart>\s*)<Placement[^>]+>' % ENV_AGENT_NAMES[0],
                         r'\1<Placement x="%g" y="4" z="%g" pitch="30" yaw="%g"/>' %
                         (pos[1][0], pos[1][1], np.random.choice(PigChaseEnvironment.VALID_YAW)), xml)
            xml = re.sub(r'(<Name>%s</Name>\s*<AgentStart>\s*)<Placement[^>]+>' % ENV_AGENT_NAMES[1],
                         r'\1<Placement x="%g" y="4" z="%g" pitch="30" yaw="%g"/>' %
                         (pos[2][0], pos[2][1], np.random.choice(PigChaseEnvironment.VALID_YAW)), xml)

        return MissionSpec(xml, True)

    def _get_pos_dist(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def reset(self, agent_type = None, agent_positions = None):
        """ Overrides reset() to allow changes in agent appearance between
        missions."""
        if agent_type and agent_type != self._agent_type or \
                self._randomize_positions:
            self._agent_type = agent_type
            self._mission = self._construct_mission()
        return super(PigChaseEnvironment, self).reset()

    def do(self, action):
        """
        Do the action
        """
        state, reward, done = super(PigChaseEnvironment, self).do(action)
        return state, reward, self.done

    def is_valid(self, world_state):
        """ Pig Chase Environment is valid if the the board and entities are present """

        if super(PigChaseEnvironment, self).is_valid(world_state):
            obs = json.loads(world_state.observations[-1].text)

            # Check we have entities
            return (ENV_ENTITIES in obs) and (ENV_BOARD in obs)
        return False
