import os

from ai_challenge.tasks.pig_chase.environment import CustomStateBuilder, ENV_AGENT_NAMES, \
    ENV_ACTIONS, ConcatStateBuilder
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.tasks.pig_chase.environment.evaluation import PigChaseEvaluator
from ai_challenge.utils import get_results_path
from ai_challenge.experiments import load_wrap_vb_learner
from ai_challenge.visualization import fit_dim_red

if __name__ == '__main__':
    clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
    path = 'simulation_DoubleDQN_DuelingNN_2017-05-19T03:15:52.26/640007'
    config_used = 'value_based_config.txt'

    saved_dir_nm, saved_learner_nm = os.path.split(path)
    agent = load_wrap_vb_learner(saved_dir_nm, saved_learner_nm, config_used,
                                 internal_to_store=[],
                                 name=ENV_AGENT_NAMES[1],
                                 nb_actions=len(ENV_ACTIONS))

    evaluation = PigChaseEvaluator(clients, agent, ConcatStateBuilder(frames_no=2, state_dim=86),
                                   os.path.join(get_results_path(), saved_dir_nm))
    evaluation.run()
