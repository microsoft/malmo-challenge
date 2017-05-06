from malmopy.agent import BaseAgent
from malmopy.visualization.visualizer import CsvVisualizer

from ai_challenge.utils import visualize_training


class CustomAgent(BaseAgent):
    def __init__(self, name, nb_actions, learner, csv_report_path):
        assert nb_actions > 0, 'Agent should at least have 1 action (got %d)' % nb_actions

        self.name = name
        self.nb_actions = nb_actions
        self._learner = learner
        self._rewards = []
        self._step_counter = 0
        self._visualizer = CsvVisualizer(output_file=csv_report_path)
        super(BaseAgent, self).__init__(self._visualizer)

    def act(self, new_state, reward, done, is_training=False):
        self._step_counter += 1
        self._rewards.append(reward)
        if done:
            self._learner.stop_episode()
            visualize_training(self._visualizer, step=self._step_counter, rewards=self._rewards)
            for name, value in self._learner.get_statistics():
                visualize_training(self._visualizer, step=self._step_counter, name=value)
            self._rewards = []
            self._step_counter = 0
        else:
            self._learner.act(new_state)

    def save(self, out_dir):
        self._learner.save(out_dir=out_dir)

    def load(self, out_dir):
        self._learner.load(out_dir=out_dir)

    def inject_summaries(self, idx):
        pass
