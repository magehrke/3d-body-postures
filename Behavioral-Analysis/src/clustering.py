from typing import Tuple

from numpy import ndarray
from sklearn import mixture
import pickle
import numpy as np


def average_over_viewpoints(data) -> Tuple[ndarray, ndarray]:
    pose_names = []
    avg_data = []
    for uparam in data:
        # Take average of each stimuli including all answers for each viewpoints
        for pose_name in poss_data[uparam]:
            avg_data.append(np.mean(poss_data[uparam][pose_name]['raw']))
            pose_names.append(pose_name)
    return np.array(pose_names), np.array(avg_data)


# Get data
with open(f'../output/all/stat_dicts/possibility_dict.pkl', "rb") as input_file:
    poss_data = pickle.load(input_file)
with open(f'../output/all/stat_dicts/realism_dict.pkl', "rb") as input_file:
    real_data = pickle.load(input_file)

poss_posenames, poss_avg_data = average_over_viewpoints(poss_data)
real_posenames, real_avg_data = average_over_viewpoints(real_data)


# Calculate GMMs
print(np.sort(poss_avg_data))
gmm_data = np.array(poss_avg_data).reshape(-1, 1)
gmm = mixture.GaussianMixture(n_components=2).fit(gmm_data)
poss_means = gmm.means_
print(poss_means)
print(gmm.covariances_)

