import numpy as np
from chainerrl import q_functions

from ai_challenge.utils import train_value_based
from ai_challenge.models import RecNNQFunc, NNQFunc
from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, CustomStateBuilder, \
    PigChaseSymbolicStateBuilder, ENV_CAUGHT_REWARD
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent

BUFFER_SIZE = 10 ** 6
EPISODIC_BUFFER_SIZE = 5 * 10 ** 3
SMALL_STEP_NUM = 10100
MED_STEP_NUM = 100100
LAR_STEP_NUM = 200100
EVAL_NO = 40
EVAL_FREQ = 2000


def dqn_exp(clients):
    model_cfg = {"gpu": -1,
                 "gamma": 1.,
                 "replay_start_size": 16,
                 "minibatch_size": 16,
                 "target_update_frequency": 500,
                 "n_times_update": 1,
                 "update_frequency": 1,
                 "episodic_update": False}

    explorer_cfg = {"start_epsilon": 0.2, "end_epsilon": 0.0,
                    "decay_steps": LAR_STEP_NUM}

    experiment_cfg = {"steps": LAR_STEP_NUM,
                      "eval_n_runs": EVAL_NO,
                      "eval_frequency": EVAL_FREQ,
                      "outdir": "results/rec_dqn_vs_focused",
                      "max_episode_len": 25}

    q_func = q_functions.SingleModelStateQFunctionWithDiscreteAction(
        NNQFunc(output_dim=3, input_dim=80, hidden_units=200))

    opponent = PigChaseChallengeAgent(name="Agent_1")
    agent_st_build = CustomStateBuilder()
    opponent_st_build = PigChaseSymbolicStateBuilder()
    opponent_env = PigChaseEnvironment(remotes=clients, state_builder=opponent_st_build, role=0,
                                       randomize_positions=True)

    agent_env = PigChaseEnvironment(remotes=clients, state_builder=agent_st_build, role=1,
                                    randomize_positions=True)

    train_value_based(agent_env=agent_env,
                      opponent_env=opponent_env,
                      opponent=opponent,
                      q_function=q_func,
                      actions_no=3,
                      experiment_config=experiment_cfg,
                      explorer_config=explorer_cfg,
                      feature_map=lambda x: x,
                      model_config=model_cfg,
                      model_type="DQN",
                      reward_norm=ENV_CAUGHT_REWARD)


def rec_dqn_exp(clients):
    model_cfg = {"gpu": -1,
                 "gamma": 1.,
                 "replay_start_size": 16,
                 "minibatch_size": 16,
                 "target_update_frequency": 500,
                 "n_times_update": 4,
                 "update_frequency": 200,
                 "episodic_update": True}

    explorer_cfg = {"start_epsilon": 0.2, "end_epsilon": 0.0,
                    "decay_steps": LAR_STEP_NUM}

    experiment_cfg = {"steps": LAR_STEP_NUM,
                      "eval_n_runs": EVAL_NO,
                      "eval_frequency": EVAL_FREQ,
                      "outdir": "results/rec_dqn_vs_focused",
                      "max_episode_len": 25}

    q_func = q_functions.SingleModelStateQFunctionWithDiscreteAction(
        RecNNQFunc(output_dim=3, input_dim=80, hidden_units=200, rec_dim=2))

    opponent = PigChaseChallengeAgent(name="Agent_1")
    agent_st_build = CustomStateBuilder()
    opponent_st_build = PigChaseSymbolicStateBuilder()
    opponent_env = PigChaseEnvironment(remotes=clients, state_builder=opponent_st_build, role=0,
                                       randomize_positions=True)

    agent_env = PigChaseEnvironment(remotes=clients, state_builder=agent_st_build, role=1,
                                    randomize_positions=True)

    train_value_based(agent_env=agent_env,
                      opponent_env=opponent_env,
                      opponent=opponent,
                      q_function=q_func,
                      actions_no=3,
                      experiment_config=experiment_cfg,
                      explorer_config=explorer_cfg,
                      feature_map=lambda x: x,
                      model_config=model_cfg,
                      model_type="DQN")
