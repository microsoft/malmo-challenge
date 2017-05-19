import logging
import numpy as np
import Queue

from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, ENV_BOARD, ENV_ENTITIES, \
    ENV_BOARD_SHAPE, ACTIONS_NUM, BOARD_SIZE, NAME_ENUM, ENT_NUM, BOARD_OFFSET

from ai_challenge.utils import Entity
from malmopy.environment.malmo import MalmoStateBuilder

logger = logging.getLogger(__name__)

"""
To check how coordinates in Minecraft work check out:
https://microsoft.github.io/malmo/0.21.0/Python_Examples/Tutorial.pdf
Page 4
"""


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


class __BoardRotator(object):
    def __init__(self):
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE, ENT_NUM), dtype=np.float32)
        self.right_top_corner = [(i, j) for i in range(np.ceil(BOARD_SIZE / 2.).astype(np.int32))
                                 for j in range(np.ceil(BOARD_SIZE / 2.).astype(np.int32))]

        self.one_hot_matrix = np.eye(4)

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
        return np.concatenate(
            (self._rot_one_hot_enc(ent_dict['Agent_1'].yaw),
             self._rot_one_hot_enc(ent_dict['Agent_2'].yaw),
             [pssd_stps / float(ACTIONS_NUM), done, rot_num % 2], self.state.ravel())).astype(
            np.float32)


board_rotator = __BoardRotator()


class CustomStateBuilder(MalmoStateBuilder):
    def __init__(self, entities_override=True):
        self._entities_override = bool(entities_override)

    def build(self, environment):
        assert isinstance(environment, PigChaseEnvironment), \
            'environment is not a Pig Chase Environment instance'

        world_obs = environment.world_observations
        if world_obs is None or ENV_BOARD not in world_obs:
            # the function below will deal with it
            transformed_state = board_rotator.rotate_board_map(world_obs, 0, False)
            return transformed_state

        # Generate symbolic view
        board = np.array(world_obs[ENV_BOARD], dtype=object).reshape(
            ENV_BOARD_SHAPE)
        entities = world_obs[ENV_ENTITIES]

        if self._entities_override:
            for entity in entities:
                board[int(entity['z'] + 1), int(entity['x'])] += '/' + entity['name']

        obs_from_env = (board, entities)
        action_count = environment.action_count
        done = environment.done

        try:
            transformed_state = board_rotator.rotate_board_map(obs_from_env, action_count, done)
        except Exception as e:
            logger.log(msg=e, level=logging.ERROR)
            logger.log(msg='Error in state builder. Last received obs: {}'.format(obs_from_env),
                       level=logging.DEBUG)
            raise e

        return transformed_state


class ConcatStateBuilder(CustomStateBuilder):
    def __init__(self, frames_no, state_dim, entities_override=True):
        super(ConcatStateBuilder, self).__init__(entities_override)
        self.state_queue = Queue.deque(
            iterable=[np.zeros((state_dim,), dtype=np.float32)] * frames_no, maxlen=frames_no)
        self.frames_no = frames_no
        self.state_dim = state_dim

    def build(self, environment):
        self.state_queue.append(super(ConcatStateBuilder, self).build(environment).copy())
        state = np.concatenate(self.state_queue).copy()
        if environment.done:
            self.state_queue = Queue.deque(
                iterable=[np.zeros((self.state_dim,), dtype=np.float32)] * self.frames_no,
                maxlen=self.frames_no)
        return state
