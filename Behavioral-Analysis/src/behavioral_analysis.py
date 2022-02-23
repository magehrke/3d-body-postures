import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import norm
from tqdm import tqdm
import os
import pickle
from collections import Counter


"""
    File: behavioral_analysis.py
    Author: MA Gehrke
    Date: 08.02.2022

    This class loads the data in form of arrays from 
    the hard drive and processes it to plot bar-
    and boxplots.
"""


class BehavioralAnalysis:
    def __init__(self, ba_numbers: [str]):
        """
        Class for executing statistical calculations on the data of the
        first, the second or both behavioral analyses.

        Make sure the 'Questionnaire.txt' has been downloaded from Qualtrics,
        using 'Numeric Values'.

        :param ba_number: chooses the data that will be used. Either from
            the first behavioral analysis, the second or all the data together.
        """
        self.ba_numbers = ba_numbers
        for n in ba_numbers:
            assert n in ['1', '2', 'all']
        print(f'Calculating plots for {ba_numbers}!')

        self.out_dir = {
            '1': f'../output/behavioral_analysis_1/',
            '2': f'../output/behavioral_analysis_2/',
            'all': f'../output/all/'
        }
        for v in self.out_dir.values():
            if not os.path.exists(v):
                os.makedirs(v)

        # Resolution of the plots
        self.dpi = 200

        # GET SCALES FOR EACH QUESTION/STIMULI/UPARAM
        # Load file containing which question belongs to which pose
        p_items_lst = np.load('../input/behavioral_analysis_1/item-question-association.npy', allow_pickle=True).item()
        p_items_lst = dict(p_items_lst)
        assert len(p_items_lst.keys()) == 324
        # Extract uparam names (108 names with 3 viewpoints each)
        uparam_names = [x[:x.find('Viewpoint') - 1] for x in
                        p_items_lst.keys()]
        uparam_names = set(uparam_names)
        assert len(uparam_names) == 108
        # Extract scales
        self.scale = {}
        for un in uparam_names:
            for stim in p_items_lst.keys():
                if stim.startswith(f'{un}_'):
                    self.scale[un] = stim[stim.find('scale') + 6:]

    def load_statistics(self, quest_dict: dict, ba_num: str) -> dict:
        """Load and return dictionary with the observations of the questions."""
        save_dir = f'{self.out_dir[ba_num]}stat_dicts/{quest_dict["prefix"]}_dict.pkl'
        if os.path.exists(save_dir):
            with open(save_dir, "rb") as input_file:
                stats = pickle.load(input_file)
                return stats

    # Not used
    def create_boxplots_vp(self, desc: dict, stats: dict):
        loop_desc = f'Creating {desc["prefix"]} boxplots'
        save_dir = f'boxplots-vp/{desc["prefix"]}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for uparam, dfs_of_vps in tqdm(stats.items(), loop_desc):
            y_labels = []
            for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                plt.boxplot(vp_dict['raw'], positions=[i+1])
                y_labels.append(f'{i+1} (n={vp_dict["n"]})')
            plt.title(f'{uparam} (Scale = {self.scale[uparam]})')
            plt.ylim((desc['likert_min'] - 1, desc['likert_max'] + 1))
            plt.yticks(range(desc['likert_min'], desc['likert_max'] + 1),
                       range(desc['likert_min'],  desc['likert_max'] + 1))
            plt.xticks(range(1, 4), y_labels)
            plt.xlabel('Viewpoint')
            plt.ylabel('Likert Scale')
            plt.savefig(f'{save_dir}/{uparam}_scale_{self.scale[uparam]}',
                        dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def create_boxplots(self, question: dict, only_hist=False):
        """
        TODO
        """
        for ba_num in self.get_loops(question):
            save_dir = f'{self.out_dir[ba_num]}boxplots/{question["prefix"]}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            stats = self.load_statistics(question, ba_num)

            scatter_data = []
            loop_desc = f'Creating {question["prefix"]} boxplots (BA {ba_num})'
            for uparam, dfs_of_vps in tqdm(stats.items(), loop_desc):
                # sum over viewpoints
                raw_all = []
                for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                    raw_all.extend(vp_dict['raw'])
                scatter_data.append([uparam, np.mean(raw_all)])
                if not only_hist:
                    fig, ax = plt.subplots(1, 2)
                    fig.suptitle(f'{uparam} (Scale = {self.scale[uparam]})')

                    ax[0].boxplot(raw_all, positions=[0])
                    ax[0].plot([0], np.mean(raw_all), '+', label='Mean')
                    ax[0].set_ylim((question['likert_min'] - 1, question['likert_max'] + 1))
                    ax[0].set_yticks(range(question['likert_min'], question['likert_max'] + 1))
                    ax[0].set_yticklabels(range(question['likert_min'],
                                                question['likert_max'] + 1))
                    ax[0].set_ylabel(f'{question["likert_str_min"]}     =>     '
                                     f'{question["likert_str_max"]}')
                    ax[0].tick_params(labelbottom=False, bottom=False)
                    ax[0].legend(frameon=False)

                    im = mpimg.imread(f'../input/stim_images/{uparam}_Viewpoint_2_scale_'
                                      f'{self.scale[uparam]}.png')
                    ax[1].set_title('Viewpoint 2')
                    ax[1].tick_params(left=False, labelleft=False,
                                      labelbottom=False, bottom=False)
                    ax[1].imshow(im)

                    strt = int(np.floor(np.mean(raw_all)))
                    subfolder = f'{save_dir}/{strt}-{strt+1}'
                    if not os.path.exists(subfolder):
                        os.mkdir(subfolder)
                    plt.savefig(f'{subfolder}/{uparam}_scale_{self.scale[uparam]}',
                                dpi=self.dpi, bbox_inches='tight')
                    plt.close()
            scatter_data = np.array(scatter_data)
            # Sort by value, but keep (name, value) setup
            scatter_data = scatter_data[np.argsort(scatter_data[:, 1])]
            scatter_data = np.transpose(scatter_data)
            float_value_arr = np.array(scatter_data[1], dtype=np.float)

            # Plot histogram of the posture means (incl. normal)
            plt.hist(float_value_arr, bins=15, density=True)
            plt.xlabel(f'{question["likert_str_min"]}     =>     '
                       f'{question["likert_str_max"]}')
            plt.xlim((question['likert_min'] - 1, question['likert_max'] + 1))
            plt.xticks(range(question['likert_min'], question['likert_max'] + 1),
                       range(question['likert_min'], question['likert_max'] + 1))
            mu, std = norm.fit(float_value_arr)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2,
                     label=f'mu = {round(mu, 2)}, std = {round(std, 2)}')
            plt.legend()
            plt.savefig(f'{save_dir}/{question["prefix"]}_hist',
                        dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # Print & export overview of values
            out_file = open(f'{save_dir}/{question["prefix"]}_values.txt', 'w')
            out_file.write(f'Stimuli, {question["likert_str_min"]} '
                           f'({question["likert_min"]}) => {question["likert_str_max"]}'
                           f'({question["likert_max"]})\n\n')
            for i in range(len(scatter_data[0])):
                out_file.write(f'{format(scatter_data[0][i] + " (" + self.scale[scatter_data[0][i]] + "),", " <30")}'
                               f'{round(float(scatter_data[1][i]), 2)}\n')
                if i % 10 == 9:
                    out_file.write('\n')
            out_file.close()

    def barplot(self, question: dict, hist_only=False):
        for ba_num in self.get_loops(question):
            ba_num = str(ba_num)
            save_dir = f'{self.out_dir[ba_num]}barplots/{question["prefix"]}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            stats = self.load_statistics(question, ba_num)

            hist_data = []
            loop_desc = f'Creating {question["prefix"]} barplots (BA {ba_num})'
            for uparam, dfs_of_vps in tqdm(stats.items(), loop_desc):
                raw_all = []
                for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                    raw_all.extend(vp_dict['raw'])
                count = Counter(raw_all)
                names = list(question['categories'].values())
                keys = list(question['categories'].keys())
                values = [0] * len(keys)
                for k, v in count.items():
                    ind = keys.index(k)
                    values[ind] = v
                max_ind = int(np.argmax(values))
                max_name = names[max_ind]
                hist_data.append(max_name)
                if not hist_only:
                    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
                    fig.suptitle(f'{uparam} (Scale = {self.scale[uparam]})')

                    ax[0].bar(names, values)
                    ax[0].tick_params(labelrotation=45)

                    im = mpimg.imread(f'../input/stim_images/{uparam}_Viewpoint_2_scale_'
                                      f'{self.scale[uparam]}.png')
                    ax[1].set_title('Viewpoint 2')
                    ax[1].tick_params(left=False, labelleft=False,
                                      labelbottom=False, bottom=False)
                    ax[1].imshow(im)

                    if not os.path.exists(f'{save_dir}/{max_name}'):
                        os.mkdir(f'{save_dir}/{max_name}')
                    plt.savefig(f'{save_dir}/{max_name}/{uparam}_scale_{self.scale[uparam]}',
                                dpi=self.dpi, bbox_inches='tight')
                    plt.close()
            # Histogram of max categories
            hist_count = Counter(hist_data)
            keys = (hist_count.keys())
            values = list(hist_count.values())
            # - Add labels on top of bars
            bars = range(0, len(values)+1)
            plt.bar(keys, values)
            plt.xticks(rotation=45)
            for i in range(len(values)):
                plt.annotate(str(values[i]), xy=(bars[i], values[i]), ha='center', va='bottom')
            plt.savefig(f'{save_dir}/{question["prefix"]}_bar',
                        dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def get_loops(self, question: dict):
        # Check BA's we want to calculate plots for
        quest_ba_type = question['behavioral_analysis']
        ba_for_quest = []
        for i in self.ba_numbers:
            if i == 'all':
                if 2 in quest_ba_type:
                    ba_for_quest.append(i)
            elif int(i) in quest_ba_type:
                ba_for_quest.append(i)

        return ba_for_quest


