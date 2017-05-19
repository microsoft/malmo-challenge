import Queue
import numpy as np

from ai_challenge.tasks.pig_chase.environment.extensions import board_rotator

"""
This module implements a simulator that can speed up training.
"""


class EnvSimulator(object):
    class State(object):
        def __init__(self, x, y, rot_x, rot_y):
            self.x = x
            self.y = y
            self.rot_x = rot_x
            self.rot_y = rot_y

    class GameState(object):
        def __init__(self, player_state, opponent_state, target_state):
            self.ps = player_state
            self.os = opponent_state
            self.ts = target_state

    _possible_rot = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def __init__(self, size, max_reward=25., escape_reward=5., step_reward=-1., show=False,
                 normalize_rewards=True):
        self.max_reward = 1 if normalize_rewards else max_reward
        self.step_reward = step_reward / max_reward if normalize_rewards else step_reward
        self.escape_reward = escape_reward / max_reward if normalize_rewards else escape_reward
        self.size = size
        self.show = show
        self.board = np.zeros((size, size))
        self.board[0, :] = 1
        self.board[:, 0] = 1
        self.board[size - 1, :] = 1
        self.board[:, size - 1] = 1
        self.mid_exit_coord = np.floor(size / 2).astype(np.int32)
        self.board[self.mid_exit_coord, 0] = 0
        self.board[self.mid_exit_coord, size - 1] = 0
        for i in range(2, size, 2):
            for j in range(2, size, 2):
                self.board[i, j] = 1
        self.pos_rot_matrix = np.array([[0, -1], [1, 0]])
        self.neg_rot_matrix = np.array([[0, 1], [-1, 0]])
        self.done = False
        self.full_state = np.zeros((size, size, 3))
        self.pssd_steps = 0
        self._player_state_rep = self.get_full_state
        self._opponent_state_rep = self.get_full_state
        self.reset()

    def move(self, ent_state, move):
        assert not self.done
        if int(move) == 1:
            ent_state.rot_x, ent_state.rot_y = self.neg_rot_matrix.dot(
                [ent_state.rot_x, ent_state.rot_y])
        elif int(move) == 2:
            ent_state.rot_x, ent_state.rot_y = self.pos_rot_matrix.dot(
                [ent_state.rot_x, ent_state.rot_y])
        elif int(move) == 0:
            new_x, new_y = ent_state.x + ent_state.rot_y, ent_state.y + ent_state.rot_x
            ent_state.x, ent_state.y = (new_x, new_y) if self.board[new_x, new_y] != 1 else (
                ent_state.x, ent_state.y)

    def reward(self, ent_state):
        if self.is_target_surrounded():
            return self.max_reward
        return self.escape_reward if self.escaped(ent_state) else self.step_reward

    def is_done(self):
        return self.is_target_surrounded() or self.escaped(self.gs.ps) or self.escaped(
            self.gs.os) or self.pssd_steps >= 25

    def escaped(self, ent_state):
        return (ent_state.x == self.mid_exit_coord and ent_state.y == 0) \
               or (ent_state.x == self.mid_exit_coord and ent_state.y == self.size - 1)

    def step(self, player_action, opponent_action):
        self.move(self.gs.ps, player_action)
        if self.is_done():
            return self._player_state_rep(), self._opponent_state_rep(), self.reward(
                self.gs.ps), self.reward(self.gs.os), True
        self.move(self.gs.os, opponent_action)
        if self.is_done():
            return self.gs.ps, self.gs.os, self.reward(self.gs.ps), self.reward(self.gs.os), True
        return self._player_state_rep(), self._opponent_state_rep(), self.step_reward, self.step_reward, False

    def is_target_surrounded(self):
        return all([(self.gs.ts.x + rot_vec[0], self.gs.ts.y + rot_vec[1]) == (
            self.gs.ps.x, self.gs.ps.y) or \
                    (self.gs.ts.x + rot_vec[0], self.gs.ts.y + rot_vec[1]) == (
                        self.gs.os.x, self.gs.os.y) or \
                    (self.board[self.gs.ts.x + rot_vec[0], self.gs.ts.y + rot_vec[1]] == 1) for
                    rot_vec in EnvSimulator._possible_rot])

    def show_state(self):
        self.full_board = self.board.copy()
        self.full_board[self.gs.ps.x, self.gs.ps.y] = 2
        self.full_board[self.gs.os.x, self.gs.os.y] = 3
        self.full_board[self.gs.ts.x, self.gs.ts.y] = 4
        print('Player ', self.gs.ps.rot_x, self.gs.ps.rot_y, self.gs.ps.x, self.gs.ps.y)
        print('Opponent ', self.gs.os.rot_x, self.gs.os.rot_y, self.gs.os.x, self.gs.os.y)
        print('Target ', self.gs.ts.rot_x, self.gs.ts.rot_y, self.gs.ts.x, self.gs.ts.y)
        print(self.full_board)

    def gen_random_state(self):
        states = []
        while len(states) < 3:
            x, y = 0, 0
            while self.board[x, y] == 1:
                x, y, = np.random.random_integers(1, self.size - 2), \
                        np.random.random_integers(1, self.size - 2)
                rot_x, rot_y = EnvSimulator._possible_rot[np.random.random_integers(0, 3)]
                if all([(x, y, rot_x, rot_y) != (state.x, state.y, state.rot_x, state.rot_y) for
                        state in states]):
                    states.append(EnvSimulator.State(x, y, rot_x, rot_y))
                    break
        return states

    def play(self):
        if self.show:
            self.show_state()
        _, _, r1, r2, done = self.step(int(input()), int(input()))

    def reset(self):
        self.pssd_steps = 0
        self.gs = EnvSimulator.GameState(*self.gen_random_state())
        self.done = False

    def get_full_state(self):
        self.full_state *= 0.
        self.full_state[self.gs.ps.x, self.gs.ps.y, :] = [1, 0, 0]
        self.full_state[self.gs.os.x, self.gs.os.y, :] = [0, 1, 0]
        self.full_state[self.gs.ts.x, self.gs.ts.y, :] = [0, 0, 1]
        self.trans_state = np.concatenate((self.full_state[1:-1, 1:-1, :].ravel(),
                                           [self.gs.ps.rot_x, self.gs.ps.rot_y],
                                           [self.gs.os.rot_x, self.gs.os.rot_y],
                                           [self.gs.ts.rot_x, self.gs.ts.rot_y]))
        return self.trans_state.astype(np.float32)


class MinecraftSimulator(EnvSimulator):
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
        super(MinecraftSimulator, self).__init__(*args, **kwargs)
        self._player_state_rep = self.get_minecraft_state
        self._opponent_state_rep = self.get_minecraft_state

    @staticmethod
    def generate_yaw(rot_x, rot_y):
        if [rot_x, rot_y] == [0, 1]:
            return 0.
        elif [rot_x, rot_y] == [1, 0]:
            return 270.
        elif [rot_x, rot_y] == [-1, 0]:
            return 90.
        elif [rot_x, rot_y] == [0, -1]:
            return 180.

    def get_minecraft_state(self):
        gen_board = MinecraftSimulator._obs_struct.copy()
        ent_lst = []
        for ent_nm, state in zip(['Agent_1', 'Agent_2', 'Pig'],
                                 [self.gs.os, self.gs.ps, self.gs.ts]):
            trans_z, trans_x = state.x + MinecraftSimulator._z_offset, state.y + MinecraftSimulator._x_offset
            gen_board[trans_z, trans_x] += '/' + ent_nm
            ent_lst.append({u'name': ent_nm, u'yaw': self.generate_yaw(state.rot_x, state.rot_y) \
                               , u'x': trans_x, u'z': trans_z})
        return gen_board, ent_lst


class FixedOpponentSimulator(MinecraftSimulator):
    def __init__(self, opponent, size):
        super(FixedOpponentSimulator, self).__init__(size)
        self.opponent = opponent
        self.opponent_reward = 0
        self._player_state_rep = self.get_full_state

    def step(self, player_action, *args):
        self.pssd_steps += 1
        opponent_action = self.opponent.act(self._opponent_state_rep(), self.opponent_reward,
                                            self.is_done())
        _, _, player_rew, self.opponent_reward, done = \
            super(FixedOpponentSimulator, self).step(player_action, opponent_action)
        return self._player_state_rep(), player_rew, done, None

    def reset(self):
        self.opponent_reward = 0
        super(FixedOpponentSimulator, self).reset()
        return self._player_state_rep()

    def get_rotated_state(self):
        return board_rotator.rotate_board_map(self.get_minecraft_state(), self.pssd_steps,
                                              self.done)


class FixedOpponentSimulatorMDP(FixedOpponentSimulator):
    def __init__(self, opponent, size, state_dim, frames_no=4):
        self.state_dim = state_dim
        self.frames_no = frames_no
        super(FixedOpponentSimulatorMDP, self).__init__(opponent, size)
        self.state_queue = Queue.deque(
            iterable=[np.zeros((state_dim,), dtype=np.float32)] * frames_no, maxlen=frames_no)
        self._player_state_rep = self.get_mdp_state

    def get_mdp_state(self):
        self.state_queue.append(self.get_rotated_state())
        return np.concatenate(self.state_queue)

    def reset(self):
        super(FixedOpponentSimulatorMDP, self).reset()
        self.state_queue = Queue.deque(
            iterable=[np.zeros((self.state_dim,), dtype=np.float32)] * self.frames_no,
            maxlen=self.frames_no)
        return self._player_state_rep()
