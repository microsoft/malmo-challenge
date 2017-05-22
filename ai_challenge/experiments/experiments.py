"""
Module defining the experiments that can be run from main.py and helper functions.
"""
import logging
import os
from datetime import datetime

import numpy as np
from chainer import optimizers, optimizer
from chainerrl import agents, replay_buffer, explorers
from chainerrl import q_functions, experiments
from chainerrl.optimizers import rmsprop_async

from malmopy.visualization.visualizer import CsvVisualizer

import ai_challenge.models as models
from ai_challenge.agents import LearningAgent
from ai_challenge.config import Config
from ai_challenge.environment import SingleEnvWrapper
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, CustomStateBuilder, \
    PigChaseSymbolicStateBuilder, ENV_CAUGHT_REWARD, extensions as env_simulator
from ai_challenge.utils import get_results_path, get_config_dir
from ai_challenge.visualization import fit_dim_red

logger = logging.getLogger(__name__)


def create_async_learner(cfg_name):
    """
    Creates a learner that can be used with asynchronous algorithms from chainerrl.
    :param cfg_name: type str, the name of the config
    :return: chainerrl agent specified in config
    """
    config = Config(cfg_name)
    network = getattr(models, config.get_str('BASIC', 'network'))(**config.get_section('NETWORK'))
    opt = rmsprop_async.RMSpropAsync(**config.get_section('OPTIMIZER'))
    opt.setup(network)
    opt.add_hook(optimizer.GradientClipping(threshold=config.get_float('BASIC', 'grad_clip')))
    learner = getattr(agents, config.get_str('BASIC', 'learner'))(network, opt,
                                                                  **config.get_section(
                                                                      'ALGORITHM'))
    return learner


def create_value_based_learner(cfg_name):
    """
    Creates a learner that can be used with value based algorithms from chainerrl.
    :param cfg_name: type str, the name of the config
    :return: chainerrl agent specified in config
    """
    vb_config = Config(cfg_name)
    network = getattr(models, vb_config.get_str('BASIC', 'network'))(
        **vb_config.get_section('NETWORK'))
    q_func = q_functions.SingleModelStateQFunctionWithDiscreteAction(model=network)
    opt = getattr(optimizers, vb_config.get_str('BASIC', 'optimizer'))(
        **vb_config.get_section('OPTIMIZER'))

    opt.setup(q_func)
    opt.add_hook(
        optimizer.GradientClipping(threshold=vb_config.get_float('BASIC', 'grad_clip')))
    rep_buf = replay_buffer.PrioritizedEpisodicReplayBuffer(
        capacity=vb_config.get_int('MEMORY_BUFFER', 'episodic_buffer_size'),
        wait_priority_after_sampling=vb_config.get_bool('MEMORY_BUFFER',
                                                        'wait_priority_after_sampling'))

    explorer = explorers.LinearDecayEpsilonGreedy(
        random_action_func=lambda: np.random.random_integers(0, vb_config.get_int('NETWORK',
                                                                                  'output_dim') - 1),
        **vb_config.get_section('EXPLORER'))

    try:
        learner = getattr(agents, vb_config.get_str('BASIC', 'learner'))(q_function=q_func,
                                                                         optimizer=opt,
                                                                         replay_buffer=rep_buf,
                                                                         phi=lambda x: x,
                                                                         explorer=explorer,
                                                                         **vb_config.get_section(
                                                                             'ALGORITHM'))
        if vb_config.get_str('BASIC', 'load_path'):
            learner.load(os.path.join(get_results_path(), vb_config.get_str('BASIC', 'load_path')))

    except AttributeError as e:
        logger.log(msg='Cannot find model {} in chainerrl.agents'.format(
            vb_config.get_str('BASIC', 'learner')),
            level=logging.ERROR)
        raise e

    logger.log(msg='Created learner {}'.format(learner.__class__.__name__),
               level=logging.INFO)
    logger.log(msg='Model parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in
                  vb_config.get_section('EXPERIMENT').items()])), level=logging.INFO)
    logger.log(msg='Explorer parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in
                  vb_config.get_section('EXPLORER').items()])), level=logging.INFO)

    return learner


def load_wrap_vb_learner(saved_dir_nm, saved_learner_nm, learner_cfg, nb_actions, name,
                         internal_to_store):
    """
    Loads a value based chainerrl agent (called learner) and wraps it in agent class that provides 
    the interface to use it with malmopy.
    :param saved_dir_nm: type str, the directory in which the chainerrl is saved
    :param saved_learner_nm: type str, the name of a saved chainerrl agent (usually number of steps
    after which the learner was saved)
    :param learner_cfg: type str, the name of config used to load the chainerrl agent
    :param nb_actions: type int, the number of actions that the agent can exectue
    :param name: type str, the name of an agent
    :param internal_to_store: type list, list of strings with names of attributes of model to store
    :return: chainerrl agent
    """
    learner = create_value_based_learner(learner_cfg)
    learner.load(os.path.join(get_results_path(), saved_dir_nm, saved_learner_nm))
    created_agent = LearningAgent(learner=learner,
                                  name=name,
                                  nb_actions=nb_actions,
                                  out_dir=os.path.join(get_results_path(), saved_dir_nm,
                                                       'state_data'),
                                  internal_to_store=internal_to_store)
    logger.log(msg='Loaded chainerrl agent from {}'.
               format(os.path.join(get_results_path(), saved_dir_nm, saved_learner_nm)),
               level=logging.INFO)
    return created_agent


def async_simulation(clients, passed_config):
    """
    Performs a simulation on a simulator specified in config.
    :param clients: not used, just to have uniform interface
    :param passed_config: type str, the name of the config to use
    """
    sim_config = Config(passed_config)
    experiment_cfg = sim_config.get_section('EXPERIMENT')
    results_dir = os.path.join(get_results_path(),
                               'simulation_{}_{}_{}'.format('A3C',
                                                            sim_config.get_str('BASIC', 'network'),
                                                            datetime.utcnow().isoformat()[:-4]))
    experiment_cfg["outdir"] = results_dir
    sim_config.copy_config(results_dir)
    learner = create_async_learner(passed_config)

    def make_env(process_idx, test):
        opponent = PigChaseChallengeAgent(name="Agent_1")
        return getattr(env_simulator, sim_config.get_str('BASIC', 'simulator'))(opponent=opponent,
                                                                                **sim_config.get_section(
                                                                                    'SIMULATOR'))

    logger.log(msg='Experiment parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in experiment_cfg.items()])),
        level=logging.INFO)
    logger.log(msg='Starting experiment, calling chainerrl function.',
               level=logging.INFO)
    experiments.train_agent_async(
        agent=learner, make_env=make_env, profile=True, **experiment_cfg)


def simulation(clients, passed_config):
    """
    Performs a simulation of Pig Chase game based on passed config
    :param clients: not used, just to have uniform interface
    :param passed_config: type str, the name of the config to use 
    """
    sim_config = Config(passed_config)
    experiment_cfg = sim_config.get_section('EXPERIMENT')
    results_dir = os.path.join(get_results_path(),
                               'simulation_{}_{}_{}'.format(sim_config.get_str('BASIC', 'learner'),
                                                            sim_config.get_str('BASIC', 'network'),
                                                            datetime.utcnow().isoformat()[:-4]))
    experiment_cfg["outdir"] = results_dir
    sim_config.copy_config(results_dir)
    opponent = PigChaseChallengeAgent(name="Agent_1", p_focused=0.95)
    agent_env = getattr(env_simulator, sim_config.get_str('BASIC', 'simulator'))(opponent=opponent,
                                                                                 **sim_config.get_section(
                                                                                     'SIMULATOR'))
    learner = create_value_based_learner(passed_config)
    logger.log(msg='Experiment parameters {}'.format(
        ' '.join([name + ':' + str(value) for name, value in experiment_cfg.items()])),
        level=logging.INFO)
    logger.log(msg='Starting experiment, calling chainerrl function.',
               level=logging.INFO)
    experiments.train_agent_with_evaluation(agent=learner,
                                            env=agent_env,
                                            **experiment_cfg)


def eval_simulation(clients, passed_config):
    eval_config = Config(passed_config)
    saved_dir_nm, saved_learner_nm = os.path.split(eval_config.get_str('BASIC', 'load_path'))
    opponent = PigChaseChallengeAgent(name="Agent_1",
                                      visualizer=CsvVisualizer(
                                          output_file=os.path.join(get_results_path(), saved_dir_nm,
                                                                   'challenge_agent_type.csv')))

    env = getattr(env_simulator, eval_config.get_str('BASIC', 'simulator'))(opponent=opponent,
                                                                            **eval_config.get_section(
                                                                                'SIMULATOR'))

    agent = load_wrap_vb_learner(saved_dir_nm, saved_learner_nm, passed_config,
                                 internal_to_store=['h'],
                                 name='evaluation_agent',
                                 nb_actions=eval_config.get_int('NETWORK', 'output_dim'))

    eval_episodes_num = eval_config.get_int('BASIC', 'eval_episodes')
    reward_sum = 0.
    steps_done = 0
    for i in range(1, eval_episodes_num + 1):
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.act(obs, reward, done, is_training=False)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            steps_done += 1

        agent.act(obs, reward, done, is_training=False)
        agent.learner.model.reset_state()


        print('episode: {}, reward per step {}'.format(i, 25 * reward_sum / float(steps_done)),
              'reward per episode: {}'.format(reward_sum / i))
    opponent._visualizer.close()
    agent.save_stored_stats(os.path.join(get_results_path(), saved_dir_nm,
                                         'internal_states.pickle'))
    fit_dim_red(os.path.join(saved_dir_nm, 'internal_states.pickle'), feature_nm='h', n_comp=2,
                opponent_type_fn=os.path.join(saved_dir_nm, 'challenge_agent_type.csv'))


def value_based_experiment(clients, passed_config):
    rvb_config = Config(os.path.join(get_config_dir(), passed_config))
    results_dir = os.path.join(get_results_path(),
                               'simulation_{}_{}_{}'.format(rvb_config.get_str('BASIC', 'learner'),
                                                            rvb_config.get_str('BASIC', 'network'),
                                                            datetime.utcnow().isoformat()[:-6]))
    experiment_cfg = rvb_config.get_section('EXPERIMENT')

    experiment_cfg["outdir"] = results_dir
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
        ' '.join(['{}:{}'.format(name, str(value)) for name, value in experiment_cfg.items()])),
        level=logging.INFO)

    logger.log(msg='Starting experiment, calling chainerrl function.',
               level=logging.INFO)

    experiments.train_agent_with_evaluation(agent=learner,
                                            env=env,
                                            **experiment_cfg)
