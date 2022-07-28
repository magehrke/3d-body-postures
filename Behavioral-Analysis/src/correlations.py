from typing import Tuple

from numpy import ndarray
from sklearn import mixture
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

from numpy.polynomial.polynomial import polyfit


"""
    File: create_2d_plots.py
    Author: MA Gehrke
    Date: 07.03.2022

    In this file we load the data for each viewpoint
    and look at the correlation between movement,
    realism and the ability of reconstructing a pose.
"""


def average_for_each_viewpoint(data) -> Tuple[ndarray, ndarray]:
    pose_names = []
    avg_data = []
    for uparam in data:
        # Take average of each stimuli
        for pose_name in data[uparam]:
            avg_data.append(np.mean(data[uparam][pose_name]['raw']))
            pose_names.append(pose_name)
    return np.array(pose_names), np.array(avg_data)


# Get data
with open(f'../output/all/stat_dicts/possibility_dict.pkl', "rb") as input_file:
    poss_data = pickle.load(input_file)
with open(f'../output/all/stat_dicts/realism_dict.pkl', "rb") as input_file:
    real_data = pickle.load(input_file)
with open(f'../output/all/stat_dicts/movement_dict.pkl', "rb") as input_file:
    move_data = pickle.load(input_file)

poss_posenames, poss_avg_data = average_for_each_viewpoint(poss_data)
real_posenames, real_avg_data = average_for_each_viewpoint(real_data)
move_posenames, move_avg_data = average_for_each_viewpoint(move_data)

out_dir = f'../output/all/all_viewpoints/correlations/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# 2D plot: realism & possiblity
pear = scipy.stats.pearsonr(poss_avg_data, real_avg_data)
# Plot a line
b, m = polyfit(poss_avg_data, real_avg_data, 1)
plt.plot(poss_avg_data, b + m * poss_avg_data, '-', c='tab:orange',
         label=f'r = {round(pear[0], 2)}')
# Plot data
plt.plot(poss_avg_data, real_avg_data, '+', c='tab:blue')
# Plot text & save
plt.xlabel(f'Possibility of body parts')
plt.ylabel(f'Realism')
plt.legend()
plt.savefig(out_dir + f'corr_poss_real_all_vps.png', dpi=200, bbox_inches='tight')
plt.close()

# 2d plot: realism & movement
pear = scipy.stats.pearsonr(move_avg_data, real_avg_data)
# plot a line
b, m = polyfit(move_avg_data, real_avg_data, 1)
plt.plot(move_avg_data, b + m * move_avg_data, '-', c='tab:orange',
         label=f'r = {round(pear[0], 2)}')
# plot data
plt.plot(move_avg_data, real_avg_data, '+', c='tab:blue')
# plot text & save
plt.xlabel(f'Movement')
plt.ylabel(f'Realism')
plt.legend()
plt.savefig(out_dir + f'corr_move_realism_all_vps.png', dpi=200, bbox_inches='tight')
plt.close()


# 2d plot: poss & move
pear = scipy.stats.pearsonr(poss_avg_data, move_avg_data)
# plot a line
b, m = polyfit(poss_avg_data, move_avg_data, 1)
plt.plot(poss_avg_data, b + m * poss_avg_data, '-', c='tab:orange',
         label=f'r = {round(pear[0], 2)}')
# plot data
plt.plot(poss_avg_data, move_avg_data, '+', c='tab:blue')
# plot text & save
plt.xlabel(f'Possibility of body parts')
plt.ylabel(f'Movement')
plt.legend()
plt.savefig(out_dir + f'corr_poss_move_all_vps.png', dpi=200, bbox_inches='tight')
plt.close()



