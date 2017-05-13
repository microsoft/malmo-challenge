import os
import pickle
from collections import defaultdict

from malmopy.agent import BaseAgent
from malmopy.visualization.visualizer import CsvVisualizer

from ai_challenge.utils import visualize_training, get_results_path
from ai_challenge.experiments import create_value_based_learner


class LearningAgent(BaseAgent):
    def __init__(self, name, nb_actions, learner, out_dir, internal_to_store=None,
                 store_internal_freq=1):

        if not nb_actions > 0:
            raise ValueError('Agent should at least have 1 action (got {})'.format(nb_actions))

        self.name = name
        self.nb_actions = nb_actions
        self.learner = learner
        self._step_counter = 0
        self._epi_counter = 0
        self._visualizer = CsvVisualizer(output_file=out_dir)
        self._internal_to_store = internal_to_store
        self._store_internal_freq = store_internal_freq
        self._storage_dict = defaultdict(dict)
        self._rewards = []
        super(BaseAgent, self).__init__(visualizer=self._visualizer)

    def act(self, new_state, reward, done, is_training=False):
        self._step_counter += 1
        self._rewards.append(reward)

        if is_training:
            action = self.learner.act_and_train(new_state, reward)
        else:
            action = self.learner.act(new_state)

        if self._internal_to_store is not None \
                and self._epi_counter % self._store_internal_freq == 0:
            self._storage_dict[self._epi_counter][self._step_counter] = self.get_model_state(
                self._internal_to_store)

        if done:
            print('done')
            self.learner.model.reset_state()
            self._epi_counter += 1
            self._step_counter = 0
            visualize_training(visualizer=self._visualizer, step=self._epi_counter,
                               rewards=self._rewards)
            self._rewards = []

        return action

    def save(self, out_dir):
        self.learner.save(out_dir)

    def load(self, out_dir):
        self.learner.load(out_dir)

    def save_stored_stats(self, path):
        self._visualizer.close()
        with open(path, 'wb') as handle:
            pickle.dump(self._storage_dict, handle)

    def get_model_state(self, state_nm_lst):
        values_dict = defaultdict(list)
        for state_nm in state_nm_lst:
            value = getattr(self.learner.model.model, state_nm)
            if value is not None:
                values_dict[state_nm] = value.data
            else:
                raise ValueError('Requested state {} is None')
        return values_dict

    def get_model_states_ref(self, state_nm_lst):
        ref_dict = {}
        for state_nm in state_nm_lst:
            ref = getattr(self.learner.model.model, state_nm)
            ref_dict[state_nm] = ref
        return ref_dict


def load_wrap_vb_learner(saved_dir_nm, saved_learner_nm, model_used, nb_actions, name,
                         internal_to_store):
    learner = create_value_based_learner(model_type=model_used)
    learner.load(os.path.join(get_results_path(), saved_dir_nm, saved_learner_nm))
    created_agent = LearningAgent(learner=learner,
                                  name=name,
                                  nb_actions=nb_actions,
                                  out_dir=os.path.join(get_results_path(), saved_dir_nm,
                                                       'state_data'),
                                  internal_to_store=internal_to_store)
    return created_agent
