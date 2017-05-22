import numpy as np


class EnvSimulator(object):
    """
    Base class for the simulator of the environment.
    """

    class State(object):
        """
        Helper class to wrap entity state.
        """

        def __init__(self, x, y, rot_x, rot_y):
            """
            Initialize the state. Coordinate here are defined in a standard way.
            :param x: type int, the x coordinate of entity.
            :param y: type int, the y coordinate of entity.
            :param rot_x: type int, the x coordinate of unit vector defining rotation
            :param rot_y: type int, the y coordinate of unit vector defining rotation
            """
            self.x = x
            self.y = y
            self.rot_x = rot_x
            self.rot_y = rot_y

    class GameState(object):
        """
        Class wrapping states of player, opponent and target.
        """

        def __init__(self, player_state, opponent_state, target_state):
            """
            Initialize the GameState.
            :param player_state: type State, state of the player
            :param opponent_state: type State, state of the opponent
            :param target_state: type State, state of the target
            """
            self.ps = player_state
            self.os = opponent_state
            self.ts = target_state

    # vectors defining possible rotations
    _possible_rot = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def __init__(self, size, max_reward=25., escape_reward=5., step_reward=-1., show=False,
                 normalize_rewards=True):
        """
        Initialize the simulator.
        :param size: type int, the size of the board.
        :param max_reward: type int, reward for catching the target
        :param escape_reward: type int, reward for escaping
        :param step_reward: type int, reward for step
        :param show: type bool, if true then shows the simulation
        :param normalize_rewards: type bool, if true then rewards are divided by max_reward
        """
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
        """
        Perform the one move in simulation on passed entity.
        :param ent_state: type State, the entity on which the move will be performed
        :param move: type int, enum representing move
        """
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
        """
        Calculate the reward based on entity state.
        :param ent_state: type State, the state of an entity
        :return: type int, the reward
        """
        if self.is_target_surrounded():
            return self.max_reward
        return self.escape_reward if self.escaped(ent_state) else self.step_reward

    def is_done(self):
        """
        Checks whether the episode is over.
        :return: type bool, true if the episode is over
        """
        return self.is_target_surrounded() or self.escaped(self.gs.ps) or self.escaped(
            self.gs.os) or self.pssd_steps >= 25

    def escaped(self, ent_state):
        """
        Checks whether the entity escaped from the map
        :param ent_state: type State, the state of an entity to check
        :return: type bool, true if the entity has escaped
        """
        return (ent_state.x == self.mid_exit_coord and ent_state.y == 0) \
               or (ent_state.x == self.mid_exit_coord and ent_state.y == self.size - 1)

    def step(self, player_action, opponent_action):
        """
        Performs the step of the environment based on actions of player and opponent.
        :param player_action: type int, enum representing player's action
        :param opponent_action: type int, enum representing opponent's action
        :return: type tuple, tuple of (new player_state, new_opponent_state, player_reward,
        opponent_reward, done)
        """
        self.move(self.gs.ps, player_action)
        self.pssd_steps += 1
        if self.is_done():
            return self._player_state_rep(), self._opponent_state_rep(), self.reward(
                self.gs.ps), self.reward(self.gs.os), True
        self.move(self.gs.os, opponent_action)
        if self.is_done():
            return self.gs.ps, self.gs.os, self.reward(self.gs.ps), self.reward(self.gs.os), True

        return self._player_state_rep(), self._opponent_state_rep(), self.step_reward, self.step_reward, False

    def is_target_surrounded(self):
        """
        Checks whether the target is trapped given the current state of the game.
        :return: type bool, true if the target is trapped
        """
        return all([(self.gs.ts.x + rot_vec[0], self.gs.ts.y + rot_vec[1]) == (
            self.gs.ps.x, self.gs.ps.y) or \
                    (self.gs.ts.x + rot_vec[0], self.gs.ts.y + rot_vec[1]) == (
                        self.gs.os.x, self.gs.os.y) or \
                    (self.board[self.gs.ts.x + rot_vec[0], self.gs.ts.y + rot_vec[1]] == 1) for
                    rot_vec in EnvSimulator._possible_rot])

    def show_state(self):
        """
        Prints current state of the game.
        """
        full_board = self.board.copy()
        full_board[self.gs.ps.x, self.gs.ps.y] = 2
        full_board[self.gs.os.x, self.gs.os.y] = 3
        full_board[self.gs.ts.x, self.gs.ts.y] = 4
        print('Player ', self.gs.ps.rot_x, self.gs.ps.rot_y, self.gs.ps.x, self.gs.ps.y)
        print('Opponent ', self.gs.os.rot_x, self.gs.os.rot_y, self.gs.os.x, self.gs.os.y)
        print('Target ', self.gs.ts.rot_x, self.gs.ts.rot_y, self.gs.ts.x, self.gs.ts.y)
        print(full_board)

    def generate_starting_positions(self):
        """
        Generates starting positions uniformly at random.
        :return: type list, list of three states
        """
        coordinate_lst = [(np.random.random_integers(1, self.size - 2),
                           np.random.random_integers(1, self.size - 2)) for _ in range(3)]

        while not (self._get_pos_dist(coordinate_lst[0], coordinate_lst[1]) > 1.1 and
                           self._get_pos_dist(coordinate_lst[1], coordinate_lst[2]) > 1.1 and
                           self._get_pos_dist(coordinate_lst[0], coordinate_lst[2]) > 1.1) and \
                all([self.board[coord] == 0 for coord in coordinate_lst]):
            coordinate_lst = [(np.random.random_integers(1, self.size - 2),
                               np.random.random_integers(1, self.size - 2)) for _ in range(3)]
        states = []
        for (x, y) in coordinate_lst:
            rot_x, rot_y = EnvSimulator._possible_rot[np.random.random_integers(0, 3)]
            states.append(EnvSimulator.State(x, y, rot_x, rot_y))
        return states

    @staticmethod
    def _get_pos_dist(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def play(self):
        """
        Method that can be used to play the game.
        """
        if self.show:
            self.show_state()
        _, _, r1, r2, done = self.step(int(input()), int(input()))

    def reset(self):
        """
        Resets the environment.
        """
        self.pssd_steps = 0
        self.gs = EnvSimulator.GameState(*self.generate_starting_positions())
        self.done = False

    def get_full_state(self):
        """
        Returns the full state of the game.
        :return: type numpy array, the full state of the game
        """
        self.full_state *= 0.
        self.full_state[self.gs.ps.x, self.gs.ps.y, :] = [1, 0, 0]
        self.full_state[self.gs.os.x, self.gs.os.y, :] = [0, 1, 0]
        self.full_state[self.gs.ts.x, self.gs.ts.y, :] = [0, 0, 1]
        return np.concatenate((self.full_state[1:-1, 1:-1, :].ravel(),
                               [self.gs.ps.rot_x, self.gs.ps.rot_y],
                               [self.gs.os.rot_x, self.gs.os.rot_y],
                               [self.gs.ts.rot_x, self.gs.ts.rot_y]))
