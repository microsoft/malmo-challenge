import numpy as np
from chainer import optimizers, optimizer
from chainerrl import experiments, agents, replay_buffer, explorers, q_functions

from ai_challenge.environment import EnvWrapper

BUFFER_SIZE = 10 ** 6
EPISODIC_BUFER_SIZE = 5 * 10 ** 3
SMALL_STEP_NUM = 10100
MED_STEP_NUM = 100100
LAR_STEP_NUM = 200100
EVAL_NO = 40
EVAL_FREQ = 2000


def train_value_based(opponent, q_function, agent_env, opponent_env,
                      model_config, explorer_config, experiment_config,
                      actions_no, feature_map, model_type, grad_clip=10., reward_norm=1.):
    opt = optimizers.Adam()
    opt.setup(q_function)
    opt.add_hook(optimizer.GradientClipping(grad_clip))
    rep_buf = replay_buffer.PrioritizedEpisodicReplayBuffer(BUFFER_SIZE,
                                                            wait_priority_after_sampling=False)

    explorer = explorers.LinearDecayEpsilonGreedy(
        random_action_func=lambda: np.random.random_integers(0, actions_no - 1),
        **explorer_config)

    agent = getattr(agents, model_type)(q_function=q_function,
                                        optimizer=opt,
                                        replay_buffer=rep_buf,
                                        phi=feature_map,
                                        explorer=explorer,
                                        **model_config)

    env = EnvWrapper(agent_env=agent_env, opponent_env=opponent_env, opponent=opponent,
                     reward_norm=reward_norm)
    experiments.train_agent_with_evaluation(agent=agent,
                                            env=env,
                                            **experiment_config)
