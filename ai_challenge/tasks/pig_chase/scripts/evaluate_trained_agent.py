import os

from ai_challenge.tasks.pig_chase.environment import CustomStateBuilder, ENV_AGENT_NAMES, \
    ENV_ACTIONS
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.tasks.pig_chase.environment.evaluation import PigChaseEvaluator
from ai_challenge.utils import get_results_path
from ai_challenge.experiments import load_wrap_vb_learner
from ai_challenge.visualization import fit_dim_red

if __name__ == '__main__':
    clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
    path = 'simulationRecNNQFunc/2017-05-14T01:02:17.592659/1000000_finish'
    config_used = 'rec_vb_config.txt'

    saved_dir_nm, saved_learner_nm = os.path.split(path)
    # agent = load_wrap_vb_learner(saved_dir_nm, saved_learner_nm, config_used,
    #                              internal_to_store=['h2', 'rec_h1'],
    #                              name=ENV_AGENT_NAMES[1],
    #                              nb_actions=len(ENV_ACTIONS))
    #
    # evaluation = PigChaseEvaluator(clients, agent, CustomStateBuilder(),
    #                                os.path.join(get_results_path(), saved_dir_nm))
    # evaluation.run()
    # agent.save_stored_stats(os.path.join(get_results_path(), saved_dir_nm, 'internal_data'))

    fit_dim_red(os.path.join(saved_dir_nm, 'internal_data'), 2, 'rec_h1',
                os.path.join(saved_dir_nm, 'challenge_agent_type.csv'))
    fit_dim_red(os.path.join(saved_dir_nm, 'internal_data'), 2, 'h2',
                os.path.join(saved_dir_nm, 'challenge_agent_type.csv'))
    # evaluation.save('sanity_check', 'pig_chase_results')
