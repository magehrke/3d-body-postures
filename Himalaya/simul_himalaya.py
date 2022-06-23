# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:58:44 2022

@author: G.Marrazzo
"""

import numpy as np
from pathlib import Path
import os
import mat73
from voxelwise_tutorials.io import load_hdf5_array
from sklearn.model_selection import check_cv
from voxelwise_tutorials.utils import generate_leave_one_run_out
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from sklearn import set_config
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.scoring import r2_score_split
from himalaya.scoring import r2_score
from himalaya.scoring import correlation_score_split
from himalaya.scoring import correlation_score
from matplotlib import pyplot as plt
import scipy.io
from himalaya.viz import plot_alphas_diagnostic

data_folder = Path("/home/magehrke/data/")
model_folder = os.path.join(data_folder, 'Models')
subj_folder = os.path.join(data_folder, 'Main_effect')
res_dir = os.path.join(data_folder, "himalaya")
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

data_suffix = "_data_python.mat"
mod_suffix = "_data_python_transform_0_6Fold_cv1_standard.h5"
subj = ["himalaya_fake_Sit_gabor_kp2d"]
#feature_names = ['VAE_dec', 'VAE_enc', 'VAEparam', "gabor", "kp2d", "kp3d"]
feature_names = ['S_it_output', 'gabor_opt6', 'kp2d']
print(f'Models: {feature_names}')

res_mod_str = '_'.join([feats for feats in feature_names])

hrf_tmp = mat73.loadmat(os.path.join(model_folder, 'hrf.mat'))
hrf = np.asarray(hrf_tmp['hrf'])

for s in subj:
    subj_filename = f'{s}{data_suffix}'
    subj_file_path = os.path.join(subj_folder, subj_filename)
    subj_data = mat73.loadmat(subj_file_path)

    cv_scores_split, cv_scores, corr_tot, corr_scores_split = [], [], [], []

    for f in range(1):
        Ytrain_cv = np.asarray(subj_data['Ytrain'][f]).squeeze()
        Ytest_cv = np.asarray(subj_data['Ytest'][f]).squeeze()

        # Center and scale targets
        # TODO: center and scale?
        Ytrain_cv -= Ytrain_cv.mean(0)
        #Ytrain_cv /= Ytrain_cv.std(0)
        Ytest_cv -= Ytest_cv.mean(0)
        # Ytest_cv /= Ytest_cv.std(0)

        # Load feature spaces
        convolve_hrf = True  # Turn this on if you wish to apply hrf beforehand
        Xs_train = []
        Xs_test = []
        n_features_list = []
        for feature_space in feature_names:
            string_mod = feature_space + mod_suffix
            file_name = os.path.join(res_dir, string_mod)
            Xi_train = (load_hdf5_array(file_name, key="X_train")).T
            if convolve_hrf:
                for i in range(Xi_train.shape[1]):
                    Xi_train[:, i] = np.convolve(Xi_train[:, i], hrf)[0:Xi_train.shape[0]]
            Xi_test = (load_hdf5_array(file_name, key="X_test")).T
            if convolve_hrf:
                for i in range(Xi_test.shape[1]):
                    Xi_test[:, i] = np.convolve(Xi_test[:, i], hrf)[0:Xi_test.shape[0]]
            Xs_train.append(Xi_train.astype(dtype="float32"))
            Xs_test.append(Xi_test.astype(dtype="float32"))
            n_features_list.append(Xi_train.shape[1])

        # Concatenate the feature spaces
        X_train = np.concatenate(Xs_train, 1)
        X_test = np.concatenate(Xs_test, 1)

        # Indices of first sample of each run
        file_name = os.path.join(model_folder, 'training_runs_onsets_6Fold.mat')
        run_onsets = mat73.loadmat(file_name)
        run_onsets = run_onsets['onset'][0]
        run_onsets = np.array(run_onsets, dtype=int)
        print(f'Run Onsets: {run_onsets}')

        # Define a leave-one-run-out cross-validation split scheme
        n_samples_train = X_train.shape[0]
        cv = generate_leave_one_run_out(n_samples_train, run_onsets)
        cv = check_cv(cv)  # copy the cross-validation splitter into a reusable
        # print(cv)

        # -------------------- DEFINE THE MODEL -------------------- #
        backend = set_backend("torch_cuda", on_error="warn")
        solver = "random_search"

        # -------------------- DOCSTRING RANDOM SEARCH -------------------- #
        # We can check its specific parameters in the function docstring
        print_docstring_rs = False
        if print_docstring_rs:
            solver_function = MultipleKernelRidgeCV.ALL_SOLVERS[solver]
            print("Docstring of the function %s:" % solver_function.__name__)
            print(solver_function.__doc__)

        ###############################################################################
        # The hyperparameter random-search solver separates the hyperparameters into a
        # shared regularization ``alpha`` and a vector of positive kernel weights which
        # sum to one. This separation of hyperparameters allows to explore efficiently
        # a large grid of values for ``alpha`` for each sampled kernel weights vector.

        n_iter = 20  # Higher = more random search = better results
        alphas = np.logspace(-10, 10, 50)
        print(f'Alphas: min = {min(alphas)}, max = {max(alphas)}, n = {len(alphas)}')

        n_targets_batch = None  # NOTE: Higher = faster, but might crash
        n_alphas_batch = None
        n_targets_batch_refit = None

        ###############################################################################
        # We put all these parameters in a dictionary ``solver_params``, and define
        # the main estimator ``MultipleKernelRidgeCV``.

        solver_params = dict(n_iter=n_iter, alphas=alphas,
                             n_targets_batch=n_targets_batch,
                             n_alphas_batch=n_alphas_batch,
                             n_targets_batch_refit=n_targets_batch_refit)

        model_1 = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
                                        solver_params=solver_params, random_state=42, cv=cv)

        # ------------------------- CREATE PIPELINE ------------------------- #
        set_config(display='diagram')

        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=False, with_std=False),  # NOTE: can be false
            Kernelizer(kernel="linear"),  # NOTE: Does not have to be linear
        )

        # ------------------------- COLUMN KERNELIZER ------------------------- #
        # The column kernelizer applies a different pipeline on each selection of
        # features, here defined with ``slices``.

        # Find the start and end of each feature space in the concatenated ``X_train``.
        start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
        slices = [
            slice(start, end)
            for start, end in zip(start_and_end[:-1], start_and_end[1:])
        ]
        print(f'Slices: {slices}')

        kernelizers_tuples = [(name, preprocess_pipeline, slice_)
                              for name, slice_ in zip(feature_names, slices)]
        column_kernelizer = ColumnKernelizer(kernelizers_tuples)
        # print(column_kernelizer)

        # Then we can define the model pipeline.
        pipe_1 = make_pipeline(column_kernelizer, model_1)

        # ------------------------- FIT RS MODEL ------------------------- #

        pipe_1.fit(X_train, Ytrain_cv)
        scores_rs = pipe_1.score(X_test, Ytest_cv)
        scores_rs = backend.to_numpy(scores_rs)

        # ------------------------- GRADIENT DECENT  ------------------------- #
        solver = "hyper_gradient"
        solver_params = dict(max_iter=200,
                             hyper_gradient_method="direct",
                             max_iter_inner_hyper=1,
                             initial_deltas="here_will_go_the_previous_deltas",
                             n_targets_batch=n_targets_batch,
                             tol=1e-5)
        model_2 = MultipleKernelRidgeCV(kernels="precomputed",
                                        solver="hyper_gradient",
                                        solver_params=solver_params,
                                        cv=cv)
        pipe_2 = make_pipeline(column_kernelizer, model_2)
        pipe_2[-1].solver_params['initial_deltas'] = pipe_1[-1].deltas_[:]
        pipe_2.fit(X_train, Ytrain_cv)
        scores = pipe_2.score(X_test, Ytest_cv)
        scores = backend.to_numpy(scores)
        cv_scores.append(scores)

        Ytest_pred = pipe_1.predict(X_test)

        cv_score = backend.to_numpy(pipe_2[1].cv_scores_)
        mean_cv_scores = np.mean(cv_score, axis=1)

        # ------------------------- PLOT Gradient Decent ------------------------- #
        x_array = np.arange(1, len(mean_cv_scores) + 1)
        plt.plot(x_array, mean_cv_scores, '-o')
        plt.grid("on")
        plt.xlabel("Number of gradient iterations")
        plt.ylabel("L2 negative loss (higher is better)")
        plt.title("Convergence curve, averaged over targets")
        plt.show()

        # ------------------------- R2 CALCULATE ------------------------- #
        Y_test_pred_split = pipe_2.predict(X_test, split=True)

        # R2
        nom = np.square(np.linalg.norm(Ytest_cv - backend.to_numpy(Ytest_pred)))
        denom = np.square(np.linalg.norm(Ytest_cv - np.mean(Ytest_cv)))
        r2 = 1 - (nom / denom)
        print(f'R2: {r2}')

        # CALCULATE Split R2 scores
        split_scores = r2_score_split(Ytest_cv, Y_test_pred_split)
        split_scores = backend.to_numpy(split_scores)
        cv_scores_split.append(split_scores)
        #print("(n_kernels, n_samples_test, n_voxels_mask) =", Y_test_pred_split.shape)
        #print("(n_kernels, n_voxels_mask) =", split_scores.shape)
        #print(f'Split Scores: {split_scores}')
        print(f'Partial R2 Mean (over voxels): {np.mean(split_scores, axis=1)}')
        print(f'Partial R2 Mean (over vox in %): '
              f'{np.mean(split_scores, axis=1)/np.sum(np.mean(split_scores, axis=1))}')
        #print(f'Partial R2 Mean scaled: {np.mean(split_scores, axis=1) / n_features_list}')
        print(f'Partial R2 Var (over voxels): {np.var(split_scores, axis=1)}')

        # Plot partial R2 scores
        for kk, score in enumerate(split_scores):
            plt.hist(score, np.linspace(0, np.max(split_scores), 50), alpha=0.7,
                     label="kernel %s" % feature_names[kk])
        plt.title(r"%s Histogram of $R^2$ generalization score split between kernels" % s)
        plt.legend()
        plt.show()

        # ------------------------- CALCULATE CORRELATION  ------------------------- #
        tot_corr = correlation_score(Ytest_cv, Ytest_pred)
        tot_corr = backend.to_numpy(tot_corr)
        corr_tot.append(tot_corr)

        corr_s = correlation_score_split(Ytest_cv, Y_test_pred_split)
        corr_s = backend.to_numpy(corr_s)
        corr_scores_split.append(corr_s)

        # ------------------------- ALPHAS ------------------------- #
        best_alphas = backend.to_numpy(pipe_2[-1].deltas_)
        print(f'Mean alphas per model: {1. / np.mean(np.exp(best_alphas), axis=1)}')
        best_alphas = 1. / np.sum(np.exp(best_alphas), axis=0)
        plot_alphas_diagnostic(best_alphas=best_alphas,
                               alphas=alphas)
        plt.show()

    out_s = s + "_results_" + res_mod_str + ".mat"
    output = os.path.join(res_dir, out_s)
    print(f'Saved model under {output}')

    scipy.io.savemat(output, {'r2_scores': cv_scores, 'r2_scores_split': cv_scores_split,
                              'corr_scores': corr_tot, 'corr_scores_split': corr_scores_split})
