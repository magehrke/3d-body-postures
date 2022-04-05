import numpy as np
import pandas as pd
import scipy.io
import mat73

import matplotlib.pyplot as plt

data = scipy.io.loadmat('../data/all_params_enc_mod_12_runs.mat')
stimuli = data['stim']
kp2d = stimuli[0]['run'][0]['info'][0][0]['kp2d'][0][19]
kp2d = kp2d[:6]
kp2d = np.array(kp2d).T
print(kp2d)
plt.plot(kp2d[0], kp2d[1], 'x')
plt.gca().set_aspect('equal')
plt.show()