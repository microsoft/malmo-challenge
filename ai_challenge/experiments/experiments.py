import os
import logging
from datetime import datetime
import numpy as np

from chainer import optimizers, optimizer
from chainerrl import agents, replay_buffer, explorers, q_functions
from chainerrl import q_functions, experiments

from ai_challenge.environment import EnvWrapper
from ai_challenge.models import RecNNQFunc, NNQFunc
from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, CustomStateBuilder, \
    PigChaseSymbolicStateBuilder, ENV_CAUGHT_REWARD
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.utils import get_results_path

BUFFER_SIZE = 10 ** 6
EPISODIC_BUFFER_SIZE = 5 * 10 ** 3
SMALL_STEP_NUM = 10100
MED_STEP_NUM = 100100
LAR_STEP_NUM = 200100
EVAL_NO = 40
EVAL_FREQ = 2000
MAX_EPI_LEN = 30

PIG_STATE_DIM = 80
PIG_ACTIONS_NUM = 3

logger = logging.getLogger(__name__)


def create_learner(model_type='DQN'):
    model_cfg = {"gpu": -1,
                 "gamma": 1.,
                 "replay_start_size": 16,
                 "minibatch_size": 16,
                 "target_update_frequency": 500,
                 "n_times_update": 5,
                 "update_frequency": 200,
                 "episodic_update": True}

    explorer_cfg = {"start_epsilon": 0.2, "end_epsilon": 0.0,
                    "decay_steps": LAR_STEP_NUM}

    grad_clip = 5.

    q_func = q_functions.SingleModelStateQFunctionWithDiscreteAction(
        RecNNQFunc(output_dim=PIG_ACTIONS_NUM, input_dim=PIG_STATE_DIM, hidden_units=200,
                   rec_dim=2))

    opt = optimizers.Adam()
    opt.setup(q_func)
    opt.add_hook(optimizer.GradientClipping(grad_clip))
    rep_buf = replay_buffer.PrioritizedEpisodicReplayBuffer(EPISODIC_BUFFER_SIZE,
                                                            wait_priority_after_sampling=False)

    explorer = explorers.LinearDecayEpsilonGreedy(
        random_action_func=lambda: np.random.random_integers(0, PIG_ACTIONS_NUM - 1),
        **explorer_cfg)

    learner = getattr(agents, model_type)(q_function=q_func,
                                          optimizer=opt,
                                          replay_buffer=rep_buf,
                                          phi=lambda x: x,
                                          explorer=explorer,
                                          **model_cfg)

    logger.log(msg='Created learner {}'.format(learner.__class__.__name__), level=logging.INFO)

    return learner


def rec_dqn_exp(clients):
    experiment_cfg = {"steps": LAR_STEP_NUM,
                      "eval_n_runs": EVAL_NO,
                      "eval_frequency": EVAL_FREQ,
                      "outdir": os.path.join(get_results_path(), 'rec_dqn_exp',
                                             datetime.utcnow().isoformat()),
                      "max_episode_len": MAX_EPI_LEN}

    opponent = PigChaseChallengeAgent(name="Agent_1")
    agent_st_build = CustomStateBuilder()
    opponent_st_build = PigChaseSymbolicStateBuilder()
    opponent_env = PigChaseEnvironment(remotes=clients,
                                       state_builder=opponent_st_build,
                                       role=0,
                                       randomize_positions=True)

    agent_env = PigChaseEnvironment(remotes=clients,
                                    state_builder=agent_st_build,
                                    role=1,
                                    randomize_positions=True)

    env = EnvWrapper(agent_env=agent_env,
                     opponent_env=opponent_env,
                     opponent=opponent,
                     reward_norm=ENV_CAUGHT_REWARD)

    learner = create_learner("DQN")

    logger.log(msg='Starting experiment, now chainerrl will carry out logging.',
               level=logging.INFO)

    experiments.train_agent_with_evaluation(agent=learner,
                                            env=env,
                                            **experiment_cfg)
