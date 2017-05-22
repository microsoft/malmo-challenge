import pickle
from collections import defaultdict

from malmopy.agent import BaseAgent
from malmopy.visualization.visualizer import CsvVisualizer

from ai_challenge.utils import visualize_training


class LearningAgent(BaseAgent):
    """
    LearningAgent is supposed to wrap agent from chainerrl library and provide interface
    to run it the specific environment.
    """

    def __init__(self, name, nb_actions, learner, out_dir, internal_to_store=None,
                 store_internal_freq=1, eval_mode = True):
        """
        Initialize the LearningAgent,
        :param name: type str, name of the agent
        :param nb_actions: type int, number of actions that the agent can perform
        :param learner: agent from third party library
        :param out_dir: type str, directory to store collected data
        :param internal_to_store: type list, attributes of a model to store
        :param store_internal_freq: type int, the frequency of saving the data from model
        """

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
        self.eval_mode = eval_mode
        super(BaseAgent, self).__init__(visualizer=self._visualizer)

    def act(self, new_state, reward, done, is_training=False):
        """
        Act in the environment.
        :param new_state: a new state from the environment that will be passed ot lerner
        :param reward: type int, reward obtained in previous turn
        :param done: type bool, boolean variable indicating whether the episode is finished
        :param is_training: type bool, boolean variable indicating whether the learner is training
        """
        self._step_counter += 1
        self._rewards.append(reward)

        if not self.eval_mode and is_training:
            action = self.learner.act_and_train(new_state, reward)
        else:
            action = self.learner.act(new_state)

        if self._internal_to_store is not None \
                and self._epi_counter % self._store_internal_freq == 0:
            self._storage_dict[self._epi_counter][self._step_counter] = self.get_model_state(
                self._internal_to_store)

        if done:
            self.learner.model.reset_state()
            self._epi_counter += 1
            self._step_counter = 0
            visualize_training(visualizer=self._visualizer, step=self._epi_counter,
                               rewards=self._rewards)

            self._rewards = []

        return action

    def save(self, out_dir):
        """
        Save the learner.
        :param out_dir: type str, path to store the learner.
        :return:
        """
        self.learner.save(out_dir)

    def load(self, out_dir):
        """
        Load the learner.
        :param out_dir: type str, path to load the learner from.
        :return:
        """
        self.learner.load(out_dir)

    def save_stored_stats(self, path):
        """
        Saves data stored in visualizer.
        :param path: type str, path to store the saved data.
        """
        self._visualizer.close()
        with open(path, 'wb') as handle:
            pickle.dump(self._storage_dict, handle)

    def get_model_state(self, layers_nm_lst):
        """
        Gets the data from hidden layer of the model.
        :param layers_nm_lst: type list, list strings with layers names to store
        :return: dictionary with (name, value) for passed names of layers
        """
        values_dict = defaultdict(list)
        for layer_nm in layers_nm_lst:
            # not very nice, but this is how it is set in chainerrl
            value = getattr(self.learner.model.model, layer_nm)
            if value is not None:
                values_dict[layer_nm] = value.data
            else:
                raise ValueError('Requested state {} is None'.format(layer_nm))
        return values_dict

    def get_model_states_ref(self, layers_nm_lst):
        """
        Returns dictionary of references to hidden layers.
        :param layers_nm_lst: type list, list with names of hidden layers
        :return: dict with keys (name, reference) for passed list of references
        """
        ref_dict = {}
        for layer_nm in layers_nm_lst:
            ref = getattr(self.learner.model.model, layer_nm)
            ref_dict[layer_nm] = ref
        return ref_dict
