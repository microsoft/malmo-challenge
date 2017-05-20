import logging
import os
from datetime import datetime

import numpy as np
from chainer import optimizers, optimizer
from chainerrl import agents, replay_buffer, explorers
from chainerrl import q_functions, experiments
from chainerrl.optimizers import rmsprop_async

import ai_challenge.environment.env_simulator as env_simulator
import ai_challenge.models as models
from ai_challenge.agents import LearningAgent
from ai_challenge.config import Config
from ai_challenge.environment import SingleEnvWrapper
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.tasks.pig_chase.environment import PigChaseEnvironment, CustomStateBuilder, \
    PigChaseSymbolicStateBuilder, ENV_CAUGHT_REWARD
from ai_challenge.utils import get_results_path, get_config_dir
from malmopy.visualization.visualizer import CsvVisualizer

logger = logging.getLogger(__name__)


def create_async_learner(cfg_name):
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


def async_simulation(clients, passed_config):
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


def simulation(clients, passed_config):
    sim_config = Config(passed_config)
    experiment_cfg = sim_config.get_section('EXPERIMENT')
    results_dir = os.path.join(get_results_path(),
                               'simulation_{}_{}_{}'.format(sim_config.get_str('BASIC', 'learner'),
                                                            sim_config.get_str('BASIC', 'network'),
                                                            datetime.utcnow().isoformat()[:-4]))
    experiment_cfg["outdir"] = results_dir
    sim_config.copy_config(results_dir)
    opponent = PigChaseChallengeAgent(name="Agent_1", p_focused=1.)
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
                                 internal_to_store=[],
                                 name='evaluation_agent',
                                 nb_actions=eval_config.get_int('NETWORK','output_dim'))

    eval_episodes_num = eval_config.get_int('BASIC', 'eval_episodes')
    reward_sum = 0.
    for i in range(1, eval_episodes_num + 1):
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward

        if i % (eval_episodes_num / 10.) == 0:
            print('episode: {}'.format(i), 'Results so far: {}'.format(reward_sum / i))
    agent.save_stored_stats(os.path.join(get_results_path(), saved_dir_nm,
                                         'internal_states.pickle'))
    opponent._visualizer.close()


def load_wrap_vb_learner(saved_dir_nm, saved_learner_nm, learner_cfg, nb_actions, name,
                         internal_to_store):
    learner = create_value_based_learner(learner_cfg)
    learner.load(os.path.join(get_results_path(), saved_dir_nm, saved_learner_nm))
    created_agent = LearningAgent(learner=learner,
                                  name=name,
                                  nb_actions=nb_actions,
                                  out_dir=os.path.join(get_results_path(), saved_dir_nm,
                                                       'state_data'),
                                  internal_to_store=internal_to_store)
    return created_agent
