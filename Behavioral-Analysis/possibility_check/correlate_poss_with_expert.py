
import numpy as np
import pandas as pd
import scipy.io
from sklearn.linear_model import LinearRegression

"""
correlate_poss_with_expert.py

Created 14.04.2022
@author MA Gehrke

This script checks, how the opinions in the behavioral
analysis about possibility coincides with the expert
opinion.

Note: People in behavioral analysis just saw the stimuli
for 750ms or so, but expert for as long as they wanted.
"""

# Viewpoint conversion between degree and numeric value
vp_dict = {-45: 1, 0: 2, 45: 3}

# Import stimuli and associated features
stimuli = scipy.io.loadmat('~/data/all_params_enc_mod_12_runs.mat')['stim']
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

# Behavioral Analysis
# 2cat: the cutoff for possibility and realism is exactly at 3 (= 2 categories per feature)
# Higher is possible/real, lower is impossible/unreal
# Group_index: 0 = both high, 1 = p high & r low, 2 = p low & r high, 3 = both low
pd2cat = pd.read_csv(f'../input/possibility_check/grouping_poss_real_2categories.csv', sep=',')

# BA: Select stimuli that are high or low on POSSIBILITY
poss2cat = np.array(pd2cat[pd2cat['group_index'] < 2]['uparam'])
imposs2cat = np.array(pd2cat[pd2cat['group_index'] > 1]['uparam'])
assert poss2cat.shape[0] + imposs2cat.shape[0] == 324
print(f'Stimuli that are > 3 in possiblity (high): {len(poss2cat)}')
print(f'Stimuli that are < 3 in possiblity (low): {len(imposs2cat)}\n')

# Import expert categorization
pd_expert = pd.read_csv(f'../data/possibility_check/possibility_by_expert.csv')

# Expert: Select stimuli that are high or low on POSSIBILITY
poss_ex = np.array(pd_expert[pd_expert['possible_2d'] == 1][['uparam', 'viewpoint']])
imposs_ex = np.array(pd_expert[pd_expert['possible_2d'] == 0][['uparam', 'viewpoint']])
assert poss_ex.shape[0] + imposs_ex.shape[0] == 324
print(f'Expert possible: {poss_ex.shape[0]}')
print(f'Expert impossible: {imposs_ex.shape[0]}\n')

# Check for matches
poss_matches = []
poss_ex_imposs_ba = []
for u, v in poss_ex:
    found = False
    for i in poss2cat:
        if f'uparam_{u}_' in i and f'Viewpoint_{v}_':
            poss_matches.append(i)
            found = True
            break

    if not found:
        poss_ex_imposs_ba.append(f'Stim_uparam_{u}_Viewpoint_{v}')
print(f'Match possible: {len(poss_matches)}')
print(f'Poss expert, but impossible BA: {len(poss_ex_imposs_ba)}')

imposs_matches = []
imposs_ex_poss_ba = []
for u, v in imposs_ex:
    found = False
    for i in imposs2cat:
        if f'uparam_{u}_' in i and f'Viewpoint_{v}_':
            imposs_matches.append(i)
            found = True
            break
    if not found:
        imposs_ex_poss_ba.append(f'Stim_uparam_{u}_Viewpoint_{v}')
print(f'\nMatch impossible: {len(imposs_matches)}')
print(f'Imposs expert, but possible BA: {len(imposs_ex_poss_ba)}')




