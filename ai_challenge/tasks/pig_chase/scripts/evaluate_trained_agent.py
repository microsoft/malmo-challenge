import os

from ai_challenge.tasks.pig_chase.environment import CustomStateBuilder, ENV_AGENT_NAMES
from ai_challenge.agents import LearningAgent
from ai_challenge.utils import get_results_path
from ai_challenge.experiments import create_learner, PIG_ACTIONS_NUM
from evaluation import PigChaseEvaluator

if __name__ == '__main__':
    saved_dir_nm = 'rec_dqn_exp/2017-05-07T12:26:43.500512'
    saved_learner_nm = '20016'
    clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]

    learner = create_learner()
    learner.load(os.path.join(get_results_path(), saved_dir_nm, saved_learner_nm))
    agent = LearningAgent(learner=learner, name='Agent_2', nb_actions=PIG_ACTIONS_NUM,
                          out_dir=os.path.join(get_results_path(), saved_dir_nm, 'eval_data'),
                          internal_to_store=['h2'])

    evaluation = PigChaseEvaluator(clients, agent, agent, CustomStateBuilder())
    evaluation.run()

    agent.save_stored(os.path.join(get_results_path(), saved_dir_nm, 'internal_data'))

    evaluation.save('sanity_check', 'pig_chase_results.json')
