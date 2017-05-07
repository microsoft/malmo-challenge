import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import manifold, preprocessing
from sklearn import decomposition

from ai_challenge.utils import get_results_path


def traj_2_array(traj_dict, feature_nm):
    data = []
    traj_ind = []
    for traj_no, traj in traj_dict.items():
        for step, values_dict in traj.items():
            value = values_dict[feature_nm]
            data.append(value)
            traj_ind.append(traj_no)
    return np.concatenate(data), np.array(traj_ind)


def fit_dim_red(traj_dict_fn, n_comp, feature_nm):
    with open(os.path.join(get_results_path(), traj_dict_fn), 'rb') as handle:
        traj_dict = pickle.load(handle)
    data, traj_ind = traj_2_array(traj_dict, feature_nm)
    data_scaled = preprocessing.scale(data)
    models = ['TSNE', 'Isomap', 'PCA']
    fig, ax = plt.subplots(nrows=1, ncols=len(models))

    for dim_red, col in zip(models, ax):
        if hasattr(manifold, dim_red):
            dim_red_model = getattr(manifold, dim_red)(n_components=n_comp)
        elif hasattr(decomposition, dim_red):
            dim_red_model = getattr(decomposition, dim_red)(n_components=n_comp)
        else:
            raise ValueError(
                'Specified dimensionality reduction not found '
                'in sklearn.mainfold or sklearn.decomposition.')
        trans_data = dim_red_model.fit_transform(data_scaled)
        col.plot(trans_data[:, 0], trans_data[:, 1], '.')
        col.set_title(dim_red)

    path, f_name = os.path.split(traj_dict_fn)

    plt.savefig(os.path.join(get_results_path(), path, f_name + '_dim_red_plot.png'))