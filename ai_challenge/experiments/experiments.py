import os
import logging
import numpy as np
from threading import Thread, active_count
from datetime import datetime
from time import sleep

from chainer import optimizers, optimizer
from chainerrl import agents, replay_buffer, explorers, q_functions
from chainerrl import q_functions, experiments
import ai_challenge.models as models

from ai_challenge.config import Config
from ai_challenge.environment import SingleEnvWrapper, EnvWrapper
from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, CustomStateBuilder, \
    PigChaseSymbolicStateBuilder, ENV_CAUGHT_REWARD
from ai_challenge.environment.env_simulator import FixedOpponentSimulator
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.utils import get_results_path, get_config_dir

BUFFER_SIZE = 10 ** 6
EPISODIC_BUFFER_SIZE = 5 * 10 ** 4
SMALL_STEP_NUM = 10100
MED_STEP_NUM = 100100
LAR_STEP_NUM = 500100
EVAL_NO = 100
EVAL_FREQ = 1000
MAX_EPI_LEN = 25

PIG_STATE_DIM = 86
PIG_ACTIONS_NUM = 3

logger = logging.getLogger(__name__)


def create_value_based_learner(cfg_name):
    vb_config = Config(cfg_name)
    network = getattr(models, vb_config.get_str('BASIC', 'network'))(
        **vb_config.get_num_section('NETWORK'))
    q_func = q_functions.SingleModelStateQFunctionWithDiscreteAction(model=network)
    opt = getattr(optimizers, vb_config.get_str('OPTIMIZER', 'optimizer'))()
    opt.setup(q_func)
    opt.add_hook(
        optimizer.GradientClipping(threshold=vb_config.get_float('OPTIMIZER', 'grad_clip')))
    rep_buf = replay_buffer.PrioritizedEpisodicReplayBuffer(
        capacity=vb_config.get_int('MEMORY_BUFFER', 'episodic_buffer_size'),
        wait_priority_after_sampling=vb_config.get_bool('MEMORY_BUFFER',
                                                        'wait_priority_after_sampling'))

    explorer = explorers.LinearDecayEpsilonGreedy(
        random_action_func=lambda: np.random.random_integers(0, PIG_ACTIONS_NUM - 1),
        **vb_config.get_num_section('EXPLORER'))

    try:
        learner = getattr(agents, vb_config.get_str('BASIC', 'learner'))(q_function=q_func,
                                                                         optimizer=opt,
                                                                         replay_buffer=rep_buf,
                                                                         phi=lambda x: x,
                                                                         explorer=explorer,
                                                                         **vb_config.get_num_section(
                                                                             'ALGORITHM'))

    except AttributeError as e:
        logger.log(msg='Cannot find model {} in chainerrl.agents'.format(
            vb_config.get_str('BASIC', 'learner')),
            level=logging.ERROR)
        raise e

    logger.log(msg='Created learner {}'.format(learner.__class__.__name__),
               level=logging.INFO)
    logger.log(msg='Model parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in
                  vb_config.get_num_section('EXPERIMENT').items()])),
        level=logging.INFO)
    logger.log(msg='Explorer parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in
                  vb_config.get_num_section('EXPLORER').items()])),
        level=logging.INFO)

    return learner


def rec_value_based_exp(clients, passed_config):
    rvb_config = Config(os.path.join(get_config_dir(), passed_config))
    experiment_cfg = rvb_config.get_num_section('EXPERIMENT')
    experiment_cfg["outdir"] = os.path.join(get_results_path(),
                                            'simulation' + rvb_config.get_str('BASIC', 'network'),
                                            datetime.utcnow().isoformat())

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

    env = SingleEnvWrapper(agent_env=agent_env,
                           opponent_env=opponent_env,
                           opponent=opponent,
                           reward_norm=ENV_CAUGHT_REWARD)

    learner = create_value_based_learner(passed_config)

    logger.log(msg='Experiment parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in experiment_cfg.items()])),
        level=logging.INFO)

    logger.log(msg='Starting experiment, calling chainerrl function.',
               level=logging.INFO)

    experiments.train_agent_with_evaluation(agent=learner,
                                            env=env,
                                            **experiment_cfg)


def simulation(clients, passed_config):
    sim_config = Config(os.path.join(get_config_dir(), passed_config))
    experiment_cfg = sim_config.get_num_section('EXPERIMENT')
    experiment_cfg["outdir"] = os.path.join(get_results_path(),
                                            'simulation' + sim_config.get_str('BASIC', 'network'),
                                            datetime.utcnow().isoformat())

    opponent = PigChaseChallengeAgent(name="Agent_1")

    agent_env = FixedOpponentSimulator(opponent=opponent, size=7)

    learner = create_value_based_learner(passed_config)

    logger.log(msg='Experiment parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in experiment_cfg.items()])),
        level=logging.INFO)

    logger.log(msg='Starting experiment, calling chainerrl function.',
               level=logging.INFO)

    experiments.train_agent_with_evaluation(agent=learner,
                                            env=agent_env,
                                            **experiment_cfg)


def self_play_exp(clients, model_type):
    experiment_cfg = {"steps": LAR_STEP_NUM,
                      "eval_n_runs": EVAL_NO,
                      "eval_frequency": EVAL_FREQ,
                      "max_episode_len": MAX_EPI_LEN}

    agent_experiment_cfg = experiment_cfg.copy()
    opponent_experiment_cfg = experiment_cfg.copy()
    agent_experiment_cfg["outdir"] = os.path.join(get_results_path(),
                                                  'self_play/rec_value_based_agent_' + model_type,
                                                  datetime.utcnow().isoformat())
    opponent_experiment_cfg["outdir"] = os.path.join(get_results_path(),
                                                     'self_play/rec_value_based_opponent_'
                                                     + model_type, datetime.utcnow().isoformat())

    agent_st_build = CustomStateBuilder()
    opponent_st_build = CustomStateBuilder()

    agent_env = PigChaseEnvironment(remotes=clients,
                                    state_builder=agent_st_build,
                                    role=1,
                                    randomize_positions=True)

    opponent_env = PigChaseEnvironment(remotes=clients,
                                       state_builder=opponent_st_build,
                                       role=0,
                                       randomize_positions=True)

    agent_wrapped_env = EnvWrapper(agent_env=agent_env,
                                   reward_norm=ENV_CAUGHT_REWARD)

    opponent_wrapped_env = EnvWrapper(agent_env=opponent_env,
                                      reward_norm=ENV_CAUGHT_REWARD)

    agent_learner = create_value_based_learner(model_type)
    opponent_learner = create_value_based_learner(model_type)

    logger.log(msg='Experiment parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in experiment_cfg.items()])),
        level=logging.INFO)

    logger.log(msg='Starting experiment, creating 2 threads calling chainerrl functions.',
               level=logging.INFO)

    agent_kwargs = agent_experiment_cfg.update({'agent': agent_learner, 'env': agent_wrapped_env})
    opponent_kwargs = agent_experiment_cfg.update(
        {'opponent': opponent_learner, 'env': opponent_wrapped_env})

    opponent_thread = Thread(target=experiments.train_agent_with_evaluation, kwargs=agent_kwargs)
    agent_thread = Thread(target=experiments.train_agent_with_evaluation, kwargs=opponent_kwargs)

    opponent_thread.start()
    agent_thread.start()

    while active_count() > 2:
        sleep(0.5)
