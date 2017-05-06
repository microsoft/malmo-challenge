from time import sleep
from threading import Thread
import math
import numpy as np

from malmopy.environment.malmo import MalmoStateBuilder

from ai_challenge.pig_chase.utils import Entity, ENV_BOARD, ENV_ENTITIES, ENV_BOARD_SHAPE, \
    ACTIONS_NUM, BOARD_SIZE, NAME_ENUM, ENT_NUM, ENV_CAUGHT_REWARD
from ai_challenge.pig_chase.environment import PigChaseEnvironment


class EnvWrapper(object):
    class ExtObs(object):
        def __init__(self, obs, passed_steps, done):
            self.obs = obs
            self.passed_steps = passed_steps
            self.done = done

    _reward_normalization = 25.

    def __init__(self, agent_env, opponent_env, opponent):
        self._opponent = opponent
        self._opponent_env = opponent_env
        self._agent_env = agent_env
        opponent_thread = Thread(target=self._run_opponent)
        opponent_thread.demon = True
        opponent_thread.start()
        sleep(1)

    def _run_opponent(self):

        obs = self._opponent_env.reset()
        reward = 0
        done = False

        while True:
            # select an action
            action = self._opponent.act(obs, reward, done, is_training=True)

            # reset if needed
            if self._opponent_env.done:
                self._opponent_env.reset()

            # take a step
            obs, reward, agent_done = self._opponent_env.do(action)

    def step(self, action):
        obs, reward, done = self._agent_env.do(action)
        obs, reward, done = self.deal_with_missing_obs(obs, reward, done)
        return obs, reward / EnvWrapper._reward_normalization, done, None

    def reset(self):
        obs = self._agent_env.reset()
        obs, reward, done = self.deal_with_missing_obs(obs, 0, False)
        return obs, float(reward) / ENV_CAUGHT_REWARD, done

    def deal_with_missing_obs(self, obs, reward, done):
        if obs is None:
            obs = self._agent_env.reset()
            reward = 0
            done = False
        return obs, reward, done

    def close(self):
        self._agent_env._agent.sendCommand("quit")
        self._opponent_env._agent.sendCommand("quit")


class CustomStateBuilder(MalmoStateBuilder):
    def __init__(self, entities_override=True):
        self._entities_override = bool(entities_override)

    @staticmethod
    def transform_incoming(obs, passed_steps, done):

        ent_data_lst = obs[1]
        state_data = obs[0]
        ent_lst = []

        if obs is None:
            ent_lst.append(Entity(name='Pig', x=0, y=0, z=0, yaw=0, pitch=0))
            ent_lst.append(Entity(name='Agent_1', x=0, y=0, z=0, yaw=0, pitch=0))
            ent_lst.append(Entity(name='Agnet_2', x=0, y=0, z=0, yaw=0, pitch=0))
            return ent_lst

        for ent_obs_data in ent_data_lst:
            x_pos, z_pos = [(j, i) for i, v in enumerate(state_data) for j, k in enumerate(v) if
                            ent_obs_data['name'] in k][0]
            x_pos -= 2
            z_pos -= 2
            x_pos = int(min(x_pos, 4))
            x_pos = int(max(x_pos, 0))
            z_pos = int(min(z_pos, 4))
            z_pos = int(max(z_pos, 0))

            if ent_obs_data['name'] == 'Pig':
                ent_lst.append(Entity(name='Pig', x=x_pos, y=0, z=z_pos, yaw=0, pitch=0))
            elif ent_obs_data['name'] == 'Agent_1':
                rot = int(ent_obs_data['yaw']) // 90
                ent_lst.append(Entity(name='Agent_1', x=x_pos, y=0, z=z_pos, yaw=rot, pitch=0))
            elif ent_obs_data['name'] == 'Agent_2':
                rot = min([-360, -270, -180, -90, 0, 90, 180, 270, 360],
                          key=lambda x: abs(x - ent_obs_data["yaw"]))
                rot = (rot % 360) // 90
                ent_lst.append(Entity(name='Agent_2', x=x_pos, y=0, z=z_pos, yaw=rot, pitch=0))

        return ent_lst, passed_steps, done

    def rotated_board_map(self, obs, passed_steps, done):

        ent_lst, pssd_stps, done = self.transform_incoming(obs, passed_steps, done)
        init_state = np.zeros((BOARD_SIZE, BOARD_SIZE, ENT_NUM), dtype=np.float32)
        trans_state = np.zeros((BOARD_SIZE, BOARD_SIZE, ENT_NUM), dtype=np.float32)
        right_top_corner = [(i, j) for i in range(math.floor(BOARD_SIZE / 2.), BOARD_SIZE + 1) for j
                            in
                            range(math.floor(BOARD_SIZE / 2.), BOARD_SIZE + 1)]
        for ent in ent_lst:
            init_state[ent.x, ent.z, NAME_ENUM[ent.name] - 1] = NAME_ENUM[ent.name]

        rot_num = 0
        init_pig_x, init_pig_z, _ = np.where(init_state == NAME_ENUM['Pig'])

        while (init_pig_x[0], init_pig_z[0]) not in right_top_corner:
            init_state = np.rot90(init_state)
            rot_num += 1
            init_pig_x, init_pig_z, _ = np.where(init_state == NAME_ENUM['Pig'])

        assert rot_num < 4, 'Problem with rotating the board'

        for ent in ent_lst:
            trans_coord = np.where(init_state == NAME_ENUM[ent.name])
            trans_state[trans_coord[0], trans_coord[1], trans_coord[2]] = 1

        agent_1_init_rot = 0
        agent_2_init_rot = 0

        for ent in ent_lst:
            if ent.name == 'Agent_1':
                agent_1_init_rot = ent.yaw
            if ent.name == 'Agent_2':
                agent_2_init_rot = ent.yaw

        agent_1_trans_rot = (agent_1_init_rot + rot_num) % 4
        agent_2_trans_rot = (agent_2_init_rot + rot_num) % 4

        add_info = np.array([pssd_stps / ACTIONS_NUM, done, rot_num % 2], dtype=np.float32)
        rot_vec = np.array([agent_1_trans_rot, agent_2_trans_rot], dtype=np.float32)

        if trans_state.shape != (5, 5, 3) or trans_state.dtype != np.dtype('float32'):
            trans_state = np.zeros((BOARD_SIZE, BOARD_SIZE, ENT_NUM), dtype=np.float32)

        if rot_vec.shape != (2,) or rot_vec.dtype != np.dtype('float32'):
            rot_vec = np.array([0, 0], dtype=np.float32)

        if add_info.shape != (3,) or add_info.dtype != np.dtype('float32'):
            add_info = np.array([0, 0], dtype=np.float32)

        full_state = np.concatenate((rot_vec, add_info, trans_state.ravel()))
        return full_state

    def build(self, environment):
        assert isinstance(environment, PigChaseEnvironment), \
            'environment is not a Pig Chase Environment instance'

        world_obs = environment.world_observations
        if world_obs is None or ENV_BOARD not in world_obs:
            return None

        # Generate symbolic view
        board = np.array(world_obs[ENV_BOARD], dtype=object).reshape(
            ENV_BOARD_SHAPE)
        entities = world_obs[ENV_ENTITIES]

        if self._entities_override:
            for entity in entities:
                board[int(entity['z'] + 1), int(entity['x'])] += '/' + entity[
                    'name']

        obs_from_env = (board, entities)
        action_count = environment.action_count
        done = environment.done

        transformed_state = self.rotated_board_map(obs_from_env, action_count, done)

        return transformed_state
