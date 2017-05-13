import os

from ai_challenge.tasks.pig_chase.environment import CustomStateBuilder, ENV_AGENT_NAMES, \
    ENV_ACTIONS
from ai_challenge.tasks.pig_chase.environment.evaluation import PigChaseEvaluator
from ai_challenge.utils import get_results_path
from ai_challenge.agents import load_wrap_vb_learner

if __name__ == '__main__':
    clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]

    saved_dir_nm = "rec_value_based_DoubleDQN/2017-05-13T08:39:45.409165"
    saved_learner_nm = '430005'
    model_used = 'DoubleDQN'
    agent = load_wrap_vb_learner(saved_dir_nm, saved_learner_nm, model_used=model_used,
                                 internal_to_store=['h2', 'rec_h1'],
                                 name=ENV_AGENT_NAMES[1], nb_actions=len(ENV_ACTIONS))

    evaluation = PigChaseEvaluator(clients, agent, agent, CustomStateBuilder())
    evaluation.run()

    agent.save_stored_stats(os.path.join(get_results_path(), saved_dir_nm, 'internal_data'))

    evaluation.save('sanity_check', 'pig_chase_results.json')
