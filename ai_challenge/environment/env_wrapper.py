from time import sleep
from threading import Thread


class EnvWrapper(object):
    _reward_normalization = 25.

    def __init__(self, agent_env, opponent_env, opponent):
        self._opponent = opponent
        self._opponent_env = opponent_env
        opponent_thread = Thread(target=self._run_opponent)
        opponent_thread.demon = True
        opponent_thread.start()
        sleep(1)
        self._agent_env = agent_env

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
        return obs, float(reward) / EnvWrapper._reward_normalization, done, None

    def reset(self):
        obs = self._agent_env.reset()
        obs, _, _ = self.deal_with_missing_obs(obs, 0, False)
        return obs

    def deal_with_missing_obs(self, obs, reward, done):
        if obs is None:
            obs = self._agent_env.reset()
            reward = 0
            done = False
        return obs, reward, done

    def close(self):
        self._agent_env._agent.sendCommand("quit")
        self._opponent_env._agent.sendCommand("quit")
