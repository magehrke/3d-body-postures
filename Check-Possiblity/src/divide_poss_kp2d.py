import numpy as np
import pandas as pd
import scipy.io
from sklearn.linear_model import LinearRegression

"""
divide_poss_kp2d.py

Created 12.04.2022
@author MA Gehrke

This script loads the kp2d data as well as 
the impossible/possible groups for each stimuli 
(obtained from the behavioral analysis)
and tries to replicate this clustering by using
different ML methods. 
"""

# Viewpoint conversion between degree and numeric value
vp_dict = {-45: 1, 0: 2, 45: 3}

# Import stimuli and associated features
stimuli = scipy.io.loadmat('../data/all_params_enc_mod_12_runs.mat')['stim']
uparam, viewpoint, scale, kp2d, latent32d = [], [], [], [], []
for i in range(6):
    info_struct = stimuli[0][1]['run'][0][0]['info']
    uparam.extend(np.array(info_struct['uparam'][0]))
    viewpoint.extend(info_struct['viewpoint'][0])
    scale.extend(info_struct['scale'][0])
    # Numpy array with all 55 2D keypoints
    kp2d.extend(np.array(info_struct['kp2d'][0]))
    # 32D latent vector in the VAE to create stimuli
    latent32d.extend(np.array(info_struct['param'][0]))
uparam = np.concatenate(uparam).ravel()
viewpoint = np.concatenate(viewpoint).ravel()
scale = np.concatenate(scale).ravel()
kp2d = np.array(kp2d)
latent32d = np.squeeze(np.array(latent32d))

# 2cat: the cutoff for possibility and realism is exactly at 3
# Higher is possible/real, lower is impossible/unreal
# Group_index: 0 = both high, 1 = p high & r low, 2 = p low & r igh, 3 = both low
pd2cat = pd.read_csv(f'../data/grouping_poss_real_2categories.csv', sep=',')
poss2cat = np.array(pd2cat[pd2cat['group_index'] < 2]['uparam'])
imposs2cat = np.array(pd2cat[pd2cat['group_index'] > 1]['uparam'])
assert poss2cat.shape[0] + imposs2cat.shape[0] == 324

poss2cat_bool = []
for u, v in zip(uparam, viewpoint):
    is_possible = True
    for i in imposs2cat:
        if f'uparam_{u}_' in i and f'Viewpoint_{vp_dict[v]}_':
            is_possible = False
            break
    poss2cat_bool.append(is_possible)

# Linear Regression - Latent parameters
reg = LinearRegression().fit(latent32d, poss2cat_bool)
print(reg.score(latent32d, poss2cat_bool))
print(reg.coef_)
print(reg.intercept_)




