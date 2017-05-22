"""
This module implements CustomStateBuilder that is used to build state for agents.

To check how coordinates in Minecraft work check out:
https://microsoft.github.io/malmo/0.21.0/Python_Examples/Tutorial.pdf
Page 4
"""
import logging
import numpy as np
import Queue

from ai_challenge.tasks.pig_chase.environment.simulator import EnvSimulator
from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, ENV_BOARD, ENV_ENTITIES, \
    ENV_BOARD_SHAPE, ACTIONS_NUM, BOARD_SIZE, NAME_ENUM, ENT_NUM, BOARD_OFFSET

from ai_challenge.utils import Entity
from malmopy.environment.malmo import MalmoStateBuilder

logger = logging.getLogger(__name__)


def transform_incoming(obs, passed_steps, done):
    ent_dict = {}
    # obs stores board at first index and list of entities at second
    if obs is None or len(obs[1]) < ENT_NUM:
        ent_dict['Pig'] = (Entity(name='Pig', x=0, y=0, z=0, yaw=0, pitch=0))
        ent_dict['Agent_1'] = (Entity(name='Agent_1', x=0, y=0, z=0, yaw=0, pitch=0))
        ent_dict['Agent_2'] = (Entity(name='Agent_2', x=0, y=0, z=0, yaw=0, pitch=0))
        logger.log(msg='Received None or incomplete observation from Malmo: {}'.format(obs),
                   level=logging.WARNING)
        return ent_dict, passed_steps, done

    # the best way to figure out how it looks is to print that or log
    state_data, ent_data_lst = obs

    for ent_data in ent_data_lst:
        x_pos, z_pos = [(j, i) for i, v in enumerate(state_data) for j, k in enumerate(v) if
                        ent_data['name'] in k][0]

        # clip values to Board range 0-4
        x_pos = np.clip(x_pos - 2, 0, BOARD_SIZE - 1)
        z_pos = np.clip(z_pos - 2, 0, BOARD_SIZE - 1)

        # move to coords centered in the middle of map
        if ent_data['name'] == 'Pig':
            ent_dict['Pig'] = (
                Entity(name='Pig', x=x_pos - BOARD_OFFSET, y=0, z=z_pos - BOARD_OFFSET, yaw=0,
                       pitch=0))
        else:
            rot = min([-360, -270, -180, -90, 0, 90, 180, 270, 360],
                      key=lambda x: abs(x - ent_data['yaw']))
            ent_dict[ent_data['name']] = (
                Entity(name=ent_data['name'], x=x_pos - BOARD_OFFSET, y=0, z=z_pos - BOARD_OFFSET,
                       yaw=rot, pitch=0))

    return ent_dict, passed_steps, done


class BoardRotator(object):
    def __init__(self, rememeber_dist_num=10):
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE, ENT_NUM), dtype=np.float32)
        self.right_top_corner = [(i, j) for i in range(np.ceil(BOARD_SIZE / 2.).astype(np.int32))
                                 for j in range(np.ceil(BOARD_SIZE / 2.).astype(np.int32))]
        self.one_hot_matrix = np.eye(4)
        self.rememeber_dist_num = rememeber_dist_num
        self.remember_movements = Queue.deque([0] * self.rememeber_dist_num,
                                              maxlen=rememeber_dist_num)

    def reset(self):
        self.remember_movements = Queue.deque([0] * self.rememeber_dist_num,
                                              maxlen=self.rememeber_dist_num)

    @staticmethod
    def _rotate_90_counterclockwise(ent_dict):
        return {ent.name: Entity(name=ent.name, x=-ent.z, y=0, z=ent.x,
                                 yaw=(ent.yaw + 90) % 360,
                                 pitch=0) for ent in ent_dict.values()}

    def _rot_one_hot_enc(self, yaw):
        return self.one_hot_matrix[((yaw % 360) // 90), :]

    def rotate_board_map(self, obs, passed_steps, done):
        ent_dict, pssd_stps, done = transform_incoming(obs, passed_steps, done)
        rot_num = 0
        while (ent_dict['Pig'].x, ent_dict['Pig'].z) not in self.right_top_corner:
            ent_dict = self._rotate_90_counterclockwise(ent_dict)
            rot_num += 1
        assert rot_num < 4, 'Problem with rotating the board'
        self.state *= 0
        for ent in ent_dict.values():
            self.state[ent.x + BOARD_OFFSET, ent.z + BOARD_OFFSET, NAME_ENUM[ent.name]] = 1

        dist_tp_pig = (np.abs(ent_dict['Pig'].x - ent_dict['Agent_1'].x) + np.abs(
            ent_dict['Pig'].z - ent_dict['Agent_1'].z)) / (2 * float(BOARD_SIZE))
        self.remember_movements.append(dist_tp_pig)
        return np.concatenate(
            (self._rot_one_hot_enc(ent_dict['Agent_1'].yaw),
             self._rot_one_hot_enc(ent_dict['Agent_2'].yaw),
             self.state.ravel(),
             self.remember_movements,
             [pssd_stps / float(ACTIONS_NUM), done, rot_num % 2])).astype(np.float32)


class CustomStateBuilder(MalmoStateBuilder):
    def __init__(self, entities_override=True, remember_dist_len=10):
        self._entities_override = bool(entities_override)
        self.board_rotator = BoardRotator(remember_dist_len)

    def build(self, environment):
        assert isinstance(environment, PigChaseEnvironment), \
            'environment is not a Pig Chase Environment instance'

        world_obs = environment.world_observations
        if world_obs is None or ENV_BOARD not in world_obs:
            # the function below will deal with it
            transformed_state = self.board_rotator.rotate_board_map(world_obs, 0, False)
            return transformed_state

        board = np.array(world_obs[ENV_BOARD], dtype=object).reshape(
            ENV_BOARD_SHAPE)
        entities = world_obs[ENV_ENTITIES]

        if self._entities_override:
            for entity in entities:
                board[int(entity['z'] + 1), int(entity['x'])] += '/' + entity['name']

        obs_from_env = (board, entities)
        action_count = environment.action_count
        done = environment.done

        if environment.done:
            self.board_rotator.reset()

        try:
            transformed_state = self.board_rotator.rotate_board_map(obs_from_env, action_count,
                                                                    done)
        except Exception as e:
            logger.log(msg=e, level=logging.ERROR)
            logger.log(msg='Error in state builder. Last received obs: {}'.format(obs_from_env),
                       level=logging.DEBUG)
            raise e

        return transformed_state


class MinecraftSimulator(EnvSimulator):
    """
    Class that adds functionality to transform observation to Minecraft-like.
    """
    _obs_struct = np.array([[u'grass', u'grass', u'grass', u'grass', u'grass', u'grass',
                             u'grass', u'grass', u'grass'],
                            [u'grass', u'sand', u'sand', u'sand', u'sand', u'sand', u'sand',
                             u'sand', u'grass'],
                            [u'grass', u'sand', u'grass', u'grass', u'grass', u'grass',
                             u'grass', u'sand', u'grass'],
                            [u'sand', u'sand', u'grass', u'sand', u'grass', u'sand',
                             u'grass', u'sand', u'sand'],
                            [u'sand', u'lapis_block', u'grass', u'grass', u'grass',
                             u'grass', u'grass', u'lapis_block', u'sand'],
                            [u'sand', u'sand', u'grass', u'sand', u'grass', u'sand',
                             u'grass', u'sand', u'sand'],
                            [u'grass', u'sand', u'grass', u'grass', u'grass', u'grass',
                             u'grass', u'sand', u'grass'],
                            [u'grass', u'sand', u'sand', u'sand', u'sand', u'sand', u'sand',
                             u'sand', u'grass'],
                            [u'grass', u'grass', u'grass', u'grass', u'grass', u'grass',
                             u'grass', u'grass', u'grass']], dtype=object)

    _ent_data_struct = {u'name': u'None', u'yaw': None, u'pitch': None, u'y': None, u'x': None,
                        u'z': None}

    _z_offset = 1
    _x_offset = 1

    def __init__(self, *args, **kwargs):
        """
        Initializes the MinecraftSimulator.
        """
        super(MinecraftSimulator, self).__init__(*args, **kwargs)
        self._player_state_rep = self.get_minecraft_state
        self._opponent_state_rep = self.get_minecraft_state

    @staticmethod
    def generate_yaw(rot_x, rot_y):
        """
        Generates yaw based on rotation vector.
        :param rot_x: type int, x coordinate of rotation vector
        :param rot_y: type int, y coordinate of rotation vector
        :return: type int, generated yaw
        """
        if [rot_x, rot_y] == [0, 1]:
            return 0.
        elif [rot_x, rot_y] == [1, 0]:
            return 270.
        elif [rot_x, rot_y] == [-1, 0]:
            return 90.
        elif [rot_x, rot_y] == [0, -1]:
            return 180.

    def get_minecraft_state(self):
        """
        Returns the Minecraft-like observation.
        :return: type tuple, Minecraft-like observation
        """
        gen_board = MinecraftSimulator._obs_struct.copy()
        ent_lst = []
        for ent_nm, state in zip(['Agent_1', 'Agent_2', 'Pig'],
                                 [self.gs.os, self.gs.ps, self.gs.ts]):
            trans_z, trans_x = state.x + MinecraftSimulator._z_offset, \
                               state.y + MinecraftSimulator._x_offset
            gen_board[trans_z, trans_x] += '/' + ent_nm
            ent_lst.append({u'name': ent_nm, u'yaw': self.generate_yaw(state.rot_x, state.rot_y),
                            u'x': trans_x, u'z': trans_z})
        return gen_board, ent_lst


class FixedOpponentSimulator(MinecraftSimulator):
    """
    Simulator in which the opponent is fixed and observation is Minecraft-like.
    """

    def __init__(self, opponent, size):
        """
        Initializes the simulator.
        :param opponent: opponent to play against.
        :param size: type int, the size of the board
        """
        self.board_rotator = BoardRotator()
        super(FixedOpponentSimulator, self).__init__(size)
        self.opponent = opponent
        self.opponent_reward = 0
        self._player_state_rep = self.get_rotated_state

    def step(self, player_action, *args):
        """
        Performs the step of the environment.
        :param player_action: type int, enum representing player's action
        :return: type tuple, tuple of (new_player_state, reward, done, info)
        """
        opponent_action = self.opponent.act(self._opponent_state_rep(), self.opponent_reward,
                                            self.is_done())
        _, _, player_rew, self.opponent_reward, done = \
            super(FixedOpponentSimulator, self).step(player_action, opponent_action)
        if done:
            self.opponent.act(self._opponent_state_rep(), self.opponent_reward,
                              self.is_done())
            logging.log(
                msg='Challenge agent using: {}'.format(self.opponent.current_agent.__class__.__name__),
                level=logging.DEBUG)

        return self._player_state_rep(), player_rew, done, None

    def reset(self):
        """
        Resets the environment.
        :return: type State, new state of the player
        """
        self.opponent_reward = 0
        super(FixedOpponentSimulator, self).reset()
        self.board_rotator.reset()
        return self._player_state_rep()

    def get_rotated_state(self):
        """
        Gets the rotated state of the board.
        :return: type numpy array, rotated board.
        """
        return self.board_rotator.rotate_board_map(self.get_minecraft_state(), self.pssd_steps,
                                                   self.done)
