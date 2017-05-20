import os

from ai_challenge.experiments import load_wrap_vb_learner
from ai_challenge.tasks.pig_chase.agents import PigChaseChallengeAgent
from ai_challenge.tasks.pig_chase.environment import ENV_AGENT_NAMES, \
    ENV_ACTIONS, ConcatStateBuilder, PigChaseSymbolicStateBuilder
from ai_challenge.tasks.pig_chase.scripts.evaluation import PigChaseEvaluator
from ai_challenge.utils import get_results_path
from malmopy.visualization.visualizer import CsvVisualizer

if __name__ == '__main__':
    clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
    path = 'simulation_DoubleDQN_DuelingRecNN_2017-05-19T19:40:53.24/1000000_finish'
    config_used = 'value_based_config.txt'

    saved_dir_nm, saved_learner_nm = os.path.split(path)
    agent = load_wrap_vb_learner(saved_dir_nm, saved_learner_nm, config_used,
                                 internal_to_store=[],
                                 name=ENV_AGENT_NAMES[1],
                                 nb_actions=len(ENV_ACTIONS))
    opponent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0], visualizer=CsvVisualizer(
        output_file=os.path.join(get_results_path(), path, 'challenge_agent_type.csv')),
                                      p_focused=0.75)
    opponent_state_builder = PigChaseSymbolicStateBuilder()
    agent_state_builder = ConcatStateBuilder(frames_no=1, state_dim=86)
    evaluation = PigChaseEvaluator(clients, agent, opponent,
                                   agent_state_builder, opponent_state_builder,
                                   os.path.join(get_results_path(), saved_dir_nm),
                                   eval_episodes=1000)
    evaluation.run()
