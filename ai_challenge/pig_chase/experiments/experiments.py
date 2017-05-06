import numpy as np
from chainer import optimizers, optimizer
from chainerrl import experiments, agents, replay_buffer, explorers,  q_functions

from ai_challenge.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.pig_chase.environment import PigChaseEnvironment, EnvWrapper, CustomStateBuilder
from ai_challenge.pig_chase.models import RecNNQFunc, NNQFunc

BUFFER_SIZE = 10 ** 6
EPISODIC_BUFER_SIZE = 5 * 10 ** 3
SMALL_STEP_NUM = 10100
MED_STEP_NUM = 100100
LAR_STEP_NUM = 200100
EVAL_NO = 40
EVAL_FREQ = 2000


def train_value_based(clients, mission_xml, opponent, q_function,
                      model_config, explorer_config, experiment_config,
                      actions_no, feature_map, model_type, grad_clip=10.):
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

    agent_env = PigChaseEnvironment(remotes=clients, state_builder=CustomStateBuilder)
    opponent_env = PigChaseEnvironment(remotes=clients, state_builder=CustomStateBuilder)
    env = EnvWrapper(agent_env=agent_env, opponent_env=opponent_env, opponent=opponent)
    experiments.train_agent_with_evaluation(agent=agent,
                                            env=env,
                                            **experiment_config)


def dqn_experiment(clients):
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

    train_value_based(clients=clients,
                      mission_xml="pig_chase.xml",
                      opponent=opponent,
                      q_function=q_func,
                      actions_no=3,
                      experiment_config=experiment_cfg,
                      explorer_config=explorer_cfg,
                      feature_map=None,
                      model_config=model_cfg,
                      model_type="DQN")


def rec_dqn_experiment(clients):
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

    train_value_based(clients=clients,
                      mission_xml="pig_chase.xml",
                      opponent=opponent,
                      q_function=q_func,
                      actions_no=3,
                      experiment_config=experiment_cfg,
                      explorer_config=explorer_cfg,
                      feature_map=None,
                      model_config=model_cfg,
                      model_type="DQN")
