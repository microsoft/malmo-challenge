import logging
from time import sleep
from threading import Thread

logger = logging.getLogger(__name__)


class EnvWrapper(object):
    def __init__(self, agent_env, opponent_env, opponent, reward_norm):
        self._opponent = opponent
        self._opponent_env = opponent_env
        self._reward_norm = reward_norm
        self._agent_env = agent_env
        opponent_thread = Thread(target=self._run_opponent)
        opponent_thread.daemon = True
        opponent_thread.start()
        sleep(1)
        logger.log(msg='Initialized {}'.format(self.__class__.__name__), level=logging.INFO)

    def _run_opponent(self):

        obs = self._opponent_env.reset()
        reward = 0
        done = False
        logger.log(msg='Starting opponent thread', level=logging.INFO)

        while True:
            if self._opponent_env.done:
                logger.log(msg='Opponent resetting env', level=logging.DEBUG)
                self._opponent_env.reset()

            action = self._opponent.act(obs, reward, done, is_training=True)

            obs, reward, done = self._opponent_env.do(action)

    def step(self, action):
        obs, reward, done = self._agent_env.do(action)
        obs, reward, done = self.deal_with_missing_obs(obs, reward, done)
        return obs, float(reward) / self._reward_norm, done, None

    def reset(self):
        obs = self._agent_env.reset()
        obs, _, _ = self.deal_with_missing_obs(obs, 0, False)
        logger.log(msg='Agent resetting env', level=logging.DEBUG)
        return obs

    def deal_with_missing_obs(self, obs, reward, done):
        if obs is None:
            obs = self._agent_env.reset()
            reward = 0
            done = False
        return obs, reward, done

    def close(self):
        logger.log(msg='Quitting mission.', level=logging.INFO)
        self._agent_env._agent.sendCommand("quit")
        self._opponent_env._agent.sendCommand("quit")
