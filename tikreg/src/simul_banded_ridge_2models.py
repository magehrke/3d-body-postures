# -*- coding: utf-8 -*-

from tikreg import models
from tikreg import utils as tikutils
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('default')
import scipy.io
import mat73
import h5py
from tikreg import spatial_priors, temporal_priors
import os
from scipy import stats
from sklearn.metrics import r2_score
import numpy.matlib as mat
from pathlib import Path
import logging
import sys

"""
Created on Fri Jul  2 15:29:33 2021

@author: G.Marrazzo
"""


def setup_logger(data_folder: Path) -> None:
    logging.basicConfig(filename=data_folder / f'body_posture.log', format='%(asctime)s %(message)s',
                        encoding='utf-8', level=logging.DEBUG)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def print_and_log(message: str):
    print(message)


def plot_figures(fit_banded_polar, Yhat) -> None:
    """

    Parameters
    ----------
    fit_banded_polar
    Yhat

    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 2, 3)
    # ax.plot(fit_banded_polar['predictions'][:,0],Yhat[:,0],'o' );
    ax.plot(fit_banded_polar['predictions'][:, 0], Yhat[:, 0], 'o')
    plt.title('voxel1 predictions [X1 X2]', fontsize=10)
    #plt.show()
    plt.plot(stats.zscore(fit_banded_polar['predictions'][:, 0]))
    plt.plot(stats.zscore(Yhat[:, 0]))
    #plt.show()


data_folder = Path("../data/Main_effect")
#setup_logger(data_folder)

name = ["_data_python.mat"]
mod_suffix = "_data_python_transform_0.mat"
# Beware: Dimension Error in S14
subj = ["S13"]
model = ["kp2d", "kp3d", "gabor", "VAE_enc", "VAEparam", "VAE_dec"]
# model = ["kp3d","kp2d","gabor"]
# model = ["VAE_enc_L1","VAE_enc_L2","VAE_enc","VAE_dec_L1","VAE_dec_L2","VAE_dec","VAEparam"]

mod_ind = np.asarray([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
data3 = mat73.loadmat('../data/hrf.mat')
hrf = np.asarray(data3['hrf'])

for s in subj:
    subject_file_path = s + name[0]
    print_and_log(f'=================================================')
    print_and_log(f'Starting calculations on subject {s} ({subject_file_path})')
    subject_data = data_folder / subject_file_path
    subj_data = mat73.loadmat(subject_data)

    for mod_i in range(12, mod_ind.shape[0]):
        # Load model data
        file_mod1 = data_folder / (model[mod_ind[mod_i][0]] + mod_suffix)
        file_mod2 = data_folder / (model[mod_ind[mod_i][1]] + mod_suffix)
        data1 = mat73.loadmat(file_mod1)
        data2 = mat73.loadmat(file_mod2)
        print_and_log(f'Iteration {mod_i+1}: Using models {model[mod_ind[mod_i][0]]} & {model[mod_ind[mod_i][1]]}')

        vox = np.arange(np.asarray(subj_data['Ytrain'][0]).squeeze().shape[1])
        # vox = tvals.argsort()[-11:-1]
        # vox = np.arange(tvals.shape[0])
        nvox = vox.shape[0]
        corrtest = np.zeros((3, nvox))
        corr1 = np.zeros((3, nvox))
        corr2 = np.zeros((3, nvox))
        beta1, beta2, pred1, pred2, hyperparam = [], [], [], [], []

        # 3 rounds of cross validation
        for f in range(3):
            # Get train and test data
            Xtrain_cv_mod1 = np.asarray(data1['Xtrain'][f]).squeeze()
            Xtest_cv_mod1 = np.asarray(data1['Xtest'][f]).squeeze()
            # Ytrain_cv_mod1= data['Ytrain']
            # Ytest_cv_mod1 = data['Ytest']
            Xtrain_cv_mod2 = np.asarray(data2['Xtrain'][f]).squeeze()
            Xtest_cv_mod2 = np.asarray(data2['Xtest'][f]).squeeze()
            Ytrain_cv = np.asarray(subj_data['Ytrain'][f]).squeeze()
            Ytest_cv = np.asarray(subj_data['Ytest'][f]).squeeze()

            # num voxels, size model 1, size model 2, num of times
            p1 = Xtrain_cv_mod1.shape[1]
            p2 = Xtrain_cv_mod2.shape[1]
            ntrain = Xtrain_cv_mod1.shape[0]
            ntest = Xtest_cv_mod2.shape[0]

            mean1 = mat.repmat(np.mean(Xtrain_cv_mod1, axis=0), ntest, 1)
            std1 = mat.repmat(np.std(Xtrain_cv_mod1, axis=0), ntest, 1)
            mean2 = mat.repmat(np.mean(Xtrain_cv_mod2, axis=0), ntest, 1)
            std2 = mat.repmat(np.std(Xtrain_cv_mod2, axis=0), ntest, 1)

            X1tr = stats.zscore(Xtrain_cv_mod1)
            X2tr = stats.zscore(Xtrain_cv_mod2)
            X1te = (Xtest_cv_mod1 - mean1) / std1
            X2te = (Xtest_cv_mod2 - mean2) / std2

            # b1    = np.random.normal(0,1,[p1,nvox]);b2     = np.random.normal(0,1,[p2,nvox]);
            # b3    = np.random.normal(0,1,[p1,nvox]);b4     = np.random.normal(0,1,[p2,nvox]);
            X1tr_hrf = np.zeros(X1tr.shape)
            X2tr_hrf = np.zeros(X2tr.shape)
            X1te_hrf = np.zeros(X1te.shape)
            X2te_hrf = np.zeros(X2te.shape)

            for i in range(X1tr.shape[1]):
                X1tr_hrf[:, i] = np.convolve(X1tr[:, i], hrf)[0:ntrain]
            for i in range(X2tr.shape[1]):
                X2tr_hrf[:, i] = np.convolve(X2tr[:, i], hrf)[0:ntrain]
            for i in range(X1te.shape[1]):
                X1te_hrf[:, i] = np.convolve(X1te[:, i], hrf)[0:ntest]
            for i in range(X2te.shape[1]):
                X2te_hrf[:, i] = np.convolve(X2te[:, i], hrf)[0:ntest]

            mean1d = mat.repmat(np.mean(Ytrain_cv, axis=0), ntest, 1)
            std1d = mat.repmat(np.std(Ytrain_cv, axis=0), ntest, 1)
            Ytr = stats.zscore(Ytrain_cv)

            Yte = (Ytest_cv - mean1d) / std1d

            Ytr = Ytr[:, vox]
            Yte = Yte[:, vox]
            # % Call banded ridge
            ratios = np.logspace(-2, 2, 25)

            delays = np.asarray(1)
            temporal_prior = temporal_priors.SphericalPrior(delays=[0])  # no delays
            feat_priors1 = spatial_priors.SphericalPrior(p1, hyparams=ratios)
            feat_priors2 = spatial_priors.SphericalPrior(p2, hyparams=[1.0])
            alphas = np.logspace(0, 4, 11)
            fit_banded_polar = models.estimate_stem_wmvnp([X1tr_hrf, X2tr_hrf], Ytr, [X1te_hrf, X2te_hrf], Yte,
                                                          feature_priors=[feat_priors1, feat_priors2],
                                                          temporal_prior=temporal_prior,
                                                          ridges=alphas,
                                                          normalize_hyparams=True,
                                                          folds=(1, 4),
                                                          performance=True,
                                                          predictions=True,
                                                          weights=True)
            #                                          verbosity=0)

            # Get weigths from kernel space to the feature space
            alphas = fit_banded_polar['optima'][:, -1]
            lambda_ones = fit_banded_polar['optima'][:, 1]
            lambda_twos = fit_banded_polar['optima'][:, 2]
            kernel_weights = fit_banded_polar['weights']
            w1 = np.linalg.multi_dot([X1tr_hrf.T, kernel_weights, np.diag(alphas), np.diag(lambda_ones ** -2)])
            w2 = np.linalg.multi_dot([X2tr_hrf.T, kernel_weights, np.diag(alphas), np.diag(lambda_twos ** -2)])

            # Check that we can recover predictions for separate models
            # predictions = pred model1 + pred model2
            Yhat = np.matmul(X1te_hrf, w1) + np.matmul(X2te_hrf, w2)
            for i in range(nvox):
                Yhat[:, i] = Yhat[:, i] / alphas[i]
            Yhat1 = np.matmul(X1te_hrf, w1)
            for i in range(nvox):
                Yhat1[:, i] = Yhat1[:, i] / alphas[i]
            Yhat2 = np.matmul(X2te_hrf, w2)
            for i in range(nvox):
                Yhat2[:, i] = Yhat2[:, i] / alphas[i]
            # Xjoin  = np.concatenate((X1.T ,X2.T));Xjoin = Xjoin.T;
            # wjoin  = np.matmul(Xjoin.T,kernel_weights);
            # Yhat1  = np.matmul(Xjoin,wjoin);

            beta1.append(w1)
            beta2.append(w2)
            pred1.append(Yhat1)
            pred2.append(Yhat2)
            hyperparam.append([[lambda_ones], [lambda_twos], [alphas]])

            for i in range(Yhat.shape[1]):
                corrtest[f, i] = np.corrcoef(Yhat[:, i], Yte[:, i])[0, 1]
                corr1[f, i] = np.corrcoef(Yhat1[:, i], Yte[:, i])[0, 1]
                corr2[f, i] = np.corrcoef(Yhat2[:, i], Yte[:, i])[0, 1]

            plot_figures(fit_banded_polar, Yhat)
            print_and_log(f'Fitted banded polar: {fit_banded_polar["performance"]}')

        print_and_log(f'Corrtest: {corrtest}')
        mean_perf = np.mean(corrtest, axis=0)
        mean_perf_1 = np.mean(corr1, axis=0)
        mean_perf_2 = np.mean(corr2, axis=0)

        out_s = s + "_results_body_" + model[mod_ind[mod_i][0]] + "_" + model[mod_ind[mod_i][1]] + ".mat"
        output = data_folder / out_s

        scipy.io.savemat(output, {'mean_perf': mean_perf, 'mean_perf_1': mean_perf_1, 'mean_perf_2': mean_perf_2,
                                  'beta1': beta1, 'beta2': beta2, 'pred1': pred1, 'pred2': pred2,
                                  'hyperparams': hyperparam})


