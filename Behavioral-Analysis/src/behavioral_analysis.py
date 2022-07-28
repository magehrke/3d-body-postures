import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import norm, shapiro, normaltest, chi2, kurtosis, skew, ttest_1samp, skewtest, kurtosistest
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

        self.scale = BehavioralAnalysis.extract_scales()

    @staticmethod
    def extract_scales() -> dict:
        """Extract and return the sale for each stimuli/question."""
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
        scale = {}
        for un in uparam_names:
            for stim in p_items_lst.keys():
                if stim.startswith(f'{un}_'):
                    scale[un] = stim[stim.find('scale') + 6:]
        return scale

    def load_statistics(self, quest_dict: dict, ba_num: str) -> dict:
        """Load and return dictionary with the observations of the questions."""
        save_dir = f'{self.out_dir[ba_num]}stat_dicts/{quest_dict["prefix"]}_dict.pkl'
        if os.path.exists(save_dir):
            with open(save_dir, "rb") as input_file:
                stats = pickle.load(input_file)
                return stats

    @staticmethod
    def grouping_folder_names(likert_mean):
        """
        We want 3 groups, but in case we want only two,
        we split the middle group again.
        """
        step_size = round((5 - 1) / 3, 2)
        if likert_mean <= 1 + step_size:
            return "1,00-2,33"
        elif likert_mean <= 1 + step_size + step_size / 2:
            return "2,34-3,00"
        elif likert_mean <= 1 + 2 * step_size:
            return "3,01-3,66"
        else:
            return "3,67-5,00"

    @staticmethod
    def grouping_folder_names_2(likert_mean):
        """
        Split in 4 groups of same size
        """
        if likert_mean <= 2:
            return "1,00-2,00"
        elif likert_mean <= 3:
            return "2,01-3,00"
        elif likert_mean <= 4:
            return "3,01-4,00"
        else:
            return "4,01-5,00"

    def create_boxplots_histograms(self, question: dict, sum_viewpoints=False, only_hist=False):
        """
        TODO
        """
        for ba_num in self.get_loops(question):
            subdir = f'all_viewpoints'
            if sum_viewpoints:
                subdir = f'viewpoint_avg'
            save_dir = f'{self.out_dir[ba_num]}{subdir}/boxplots/{question["prefix"]}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            stats = self.load_statistics(question, ba_num)

            scatter_data = []
            loop_desc = f'Creating {question["prefix"]} boxplots (BA {ba_num})'
            for uparam, dfs_of_vps in tqdm(stats.items(), loop_desc):
                hist_data = []
                if sum_viewpoints:
                    # sum over viewpoints
                    raw_all = []
                    for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                        raw_all.extend(vp_dict['raw'])
                    scatter_data.append([uparam, np.mean(raw_all)])
                    hist_data.append(['proxy', raw_all])
                else:
                    for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                        hist_data.append([pose_name, vp_dict['raw']])
                        scatter_data.append([pose_name, np.mean(vp_dict['raw'])])

                if not only_hist:
                    for pose_name, hd in hist_data:
                        fig, ax = plt.subplots(1, 2)
                        if sum_viewpoints:
                            fig.suptitle(f'{uparam} (Scale = {self.scale[uparam]})')
                        else:
                            fig.suptitle(pose_name)

                        ax[0].boxplot(hd, positions=[0])
                        ax[0].plot([0], np.mean(hd), '+',
                                   label=f'Mean\n({round(np.mean(hd), 2)})')
                        ax[0].set_ylim((question['likert_min'] - 1, question['likert_max'] + 1))
                        ax[0].set_yticks(range(question['likert_min'], question['likert_max'] + 1))
                        ax[0].set_yticklabels(range(question['likert_min'],
                                                    question['likert_max'] + 1))
                        ax[0].set_ylabel(f'{question["likert_str_min"]}     =>     '
                                         f'{question["likert_str_max"]}')
                        ax[0].tick_params(labelbottom=False, bottom=False)
                        ax[0].legend(frameon=False)

                        if sum_viewpoints:
                            im = mpimg.imread(f'../input/stim_images/{uparam}_Viewpoint_2_scale_'
                                              f'{self.scale[uparam]}.png')
                            ax[1].set_title('Viewpoint 2')
                        else:
                            im = mpimg.imread(f'../input/stim_images/{pose_name}.png')
                            vi = pose_name.find("View")
                            ax[1].set_title(f'Viewpoint {pose_name[vi+10:vi+11]}')

                        ax[1].tick_params(left=False, labelleft=False,
                                          labelbottom=False, bottom=False)
                        ax[1].imshow(im)

                        final_dir = BehavioralAnalysis.grouping_folder_names(np.mean(hd))
                        subfolder = f'{save_dir}/{final_dir}'
                        if not os.path.exists(subfolder):
                            os.mkdir(subfolder)
                        save_path = f'{subfolder}/{uparam}_scale_{self.scale[uparam]}'
                        if not sum_viewpoints:
                            save_path = f'{subfolder}/{pose_name}'
                        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                        plt.close()

            scatter_data = np.array(scatter_data)
            # Sort by value, but keep (name, value) setup
            scatter_data = scatter_data[np.argsort(scatter_data[:, 1])]
            scatter_data = np.transpose(scatter_data)
            float_value_arr = np.array(scatter_data[1], dtype=np.float)

            mu = np.mean(float_value_arr)
            std = np.std(float_value_arr)

            # Plot histogram of the posture means (incl. normal)
            n_bins, bins, _ = plt.hist(float_value_arr, bins=15, density=False)
            plt.axvline(np.mean(float_value_arr), color='k', linestyle='dashed', linewidth=1, label=f'Mean = {round(mu, 2)} (SD = {round(std, 2)})')  # Mean
            plt.ylabel('Number of stimuli')
            plt.xlabel(f'{question["likert_str_min"]}     =>     '
                       f'{question["likert_str_max"]}')
            plt.xlim((question['likert_min'] - 1, question['likert_max'] + 1))
            plt.xticks(range(question['likert_min'], question['likert_max'] + 1),
                       range(question['likert_min'], question['likert_max'] + 1))

            # Plot normal distribution
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            # plt.plot(x, p, 'k', linewidth=2)
            plt.legend()
            save_path = f'{save_dir}/{question["prefix"]}_hist_all_vps'
            if sum_viewpoints:
                save_path = f'{save_dir}/{question["prefix"]}_hist_vp_avg'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # Descriptive statistics and tests
            print(f'------------------------------')
            print(f'Descriptives {question["prefix"]} (ba_num={ba_num}, sum_vps={sum_viewpoints})')
            n_bin_max = np.argmax(n_bins)
            bin_median_1 = round(bins[n_bin_max], 2)
            bin_median_2 = round(bins[n_bin_max+1], 2)
            print(f'bin median = [{bin_median_1}, {bin_median_2}]')
            kurt_x = kurtosis(float_value_arr)
            skew_x = skew(float_value_arr)
            skew_t = skewtest(float_value_arr)
            kurt_t = kurtosistest(float_value_arr)
            _, shapiro_p = shapiro(float_value_arr)
            print(f'mu = {round(mu, 2)}, std = {round(std, 2)}')
            print(f'kurt = {round(kurt_x, 2)}, skew = {round(skew_x, 2)}')
            print(f'Skewtest = {skew_t}')
            print(f'Kurtosistest = {kurt_t}')
            print(f'Shapiro p: {round(shapiro_p, 3)}')

            def var_test(x, va0, direction="lower", alpha=0.05):
                n = len(x)
                Q = (n - 1) * np.var(x) / va0
                if direction == "lower":
                    q = chi2.ppf(alpha, n - 1)
                    if Q <= q:
                        print("Var test: equal or lower")
                    else:
                        print("Var test: greater :(")
            var_test(float_value_arr, 0.44)  # 68.2% shall be between in a range of 1.33
            _, ttest_p = ttest_1samp(float_value_arr, 3)
            print(f'T-Test p: {ttest_p}')
            print(f'------------------------------')

            # Print & export overview of values
            if sum_viewpoints:
                out_file = open(f'{save_dir}/{question["prefix"]}_values_vps_avg.txt', 'w')
            else:
                out_file = open(f'{save_dir}/{question["prefix"]}_values_all_vps.txt', 'w')

            out_file.write(f'Stimuli, {question["likert_str_min"]} '
                           f'({question["likert_min"]}) => {question["likert_str_max"]}'
                           f'({question["likert_max"]})\n\n')
            last_group = BehavioralAnalysis.grouping_folder_names(float(scatter_data[1][0]))
            groups_in_total = []
            groups_in_total_2 = []
            for i in range(len(scatter_data[0])):
                if sum_viewpoints:
                    out_str = f'{format(scatter_data[0][i] + " (" + self.scale[scatter_data[1][i]] + "),", " <30")}'
                else:
                    out_str = f'{format(scatter_data[0][i] + ", ", " <43")}'

                out_str += f'{round(float(scatter_data[1][i]), 2)}\n'
                out_file.write(out_str)
                curr_group = BehavioralAnalysis.grouping_folder_names(float(scatter_data[1][i]))
                groups_in_total.append(curr_group)
                groups_in_total_2.append(BehavioralAnalysis.grouping_folder_names_2(float(scatter_data[1][i])))
                if last_group is not curr_group:
                    out_file.write('\n')
                    last_group = curr_group
            # Summary with 3 groups
            n_stim = len(groups_in_total)
            count = Counter(groups_in_total)
            out_file.write(f'\nNumber of stimuli in groups:')
            for k, v in count.items():
                out_file.write(f'\n{k}: {v} ({round(v/n_stim*100, 2)}%)')
            # Summary with 4 groups
            out_file.write('\n')
            count = Counter(groups_in_total_2)
            out_file.write(f'\nNumber of stimuli in groups:')
            for k, v in count.items():
                out_file.write(f'\n{k}: {v} ({round(v/n_stim*100, 2)}%)')
            out_file.close()


    def barplot(self, question: dict, sum_viewpoints=False, hist_only=False):
        for ba_num in self.get_loops(question):
            ba_num = str(ba_num)
            subdir = f'viewpoint_avg' if sum_viewpoints else f'all_viewpoints'
            save_dir = f'{self.out_dir[ba_num]}{subdir}/barplots/{question["prefix"]}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            stats = self.load_statistics(question, ba_num)

            hist_data = []
            loop_desc = f'Creating {question["prefix"]} barplots (BA {ba_num})'
            consensus_values = []
            for uparam, dfs_of_vps in tqdm(stats.items(), loop_desc):
                hist_uparam = []
                if sum_viewpoints:
                    raw_all = []
                    for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                        raw_all.extend(vp_dict['raw'])
                    hist_uparam.append(['proxy', raw_all])
                else:
                    for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                        hist_uparam.append([pose_name, vp_dict['raw']])

                for pose_name, hu in hist_uparam:
                    count = Counter(hu)
                    names = list(question['categories'].values())
                    keys = list(question['categories'].keys())
                    values = [0] * len(keys)
                    for k, v in count.items():
                        ind = keys.index(k)
                        values[ind] = v
                    max_ind = int(np.argmax(values))
                    max_name = names[max_ind]
                    consensus_values.append([pose_name, sorted(values), max_name])
                    hist_data.append(max_name)
                    if not hist_only:
                        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
                        if sum_viewpoints:
                            fig.suptitle(f'{uparam} (Scale = {self.scale[uparam]})')
                        else:
                            fig.suptitle(pose_name)

                        ax[0].bar(names, values)
                        ax[0].tick_params(labelrotation=45)

                        if sum_viewpoints:
                            im = mpimg.imread(f'../input/stim_images/{uparam}_Viewpoint_2_scale_'
                                              f'{self.scale[uparam]}.png')
                            ax[1].set_title('Viewpoint 2')
                        else:
                            im = mpimg.imread(f'../input/stim_images/{pose_name}.png')
                            vi = pose_name.find("View")
                            ax[1].set_title(f'Viewpoint {pose_name[vi+10:vi+11]}')

                        ax[1].tick_params(left=False, labelleft=False,
                                          labelbottom=False, bottom=False)
                        ax[1].imshow(im)

                        if not os.path.exists(f'{save_dir}/{max_name}'):
                            os.mkdir(f'{save_dir}/{max_name}')
                        save_path = f'{save_dir}/{max_name}/{uparam}_scale_{self.scale[uparam]}'
                        if not sum_viewpoints:
                            save_path = f'{save_dir}/{max_name}/{pose_name}'
                        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                        plt.close()

            # Histogram of max categories
            hist_count = Counter(hist_data)
            keys = list(hist_count.keys())
            values = list(hist_count.values())
            n_ans = np.sum(values)
            val_perc = values / n_ans * 100
            # - Add labels on top of bars
            bars = range(0, len(values)+1)
            plt.bar(keys, values)
            plt.xticks(rotation=45)
            plt.ylabel("Number of stimuli")
            for i in range(len(values)):
                plt.annotate(f'{values[i]} ({round(val_perc[i], 1)}%)', xy=(bars[i], values[i]), ha='center', va='bottom')
            suffix = 'vp-avg' if sum_viewpoints else 'all-vps'
            plt.savefig(f'{save_dir}/{question["prefix"]}_bar_{suffix}',
                        dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # ----- CONSENSUS ----- #
            # Plot consensus (how much percent ansered max category
            # Sort consensus by the intervals that got most answers
            sorted_values_list = []
            for i in range(len(keys)):
                name_of_max_category = keys[i]
                for j in range(len(consensus_values)):
                    if consensus_values[j][2] == name_of_max_category:
                        sorted_values_list.append(consensus_values[j])
            assert len(consensus_values) == 324
            # Get only the values
            cons_only_values = list(zip(*sorted_values_list))[1]
            # Turn into percentages
            cons_perc_only_max = []
            for v in cons_only_values:
                perc = max(v) / sum(v)
                perc = perc * 100
                perc = round(perc)
                cons_perc_only_max.append(perc)
            # PLOTTING
            fig, ax1 = plt.subplots(1, 1)
            ax1.scatter(range(1, 325), cons_perc_only_max, marker="x", alpha=0.7)
            # ax1.bar(range(1, 325), cons_perc_only_max)
            mu = round(np.median(cons_perc_only_max), 2)
            print(f'Consensus for {question["prefix"]} = {mu}')
            ax1.axhline(mu, linestyle='dashed', color='k', label=f'All stimuli')

            mycolors = ['orange', 'blue', 'red', 'tab:brown', 'pink']
            # Plot horizontal lines
            start_ind = 0
            for i in range(len(values)):
                med = np.median(cons_perc_only_max[start_ind:start_ind+values[i]])
                ax1.hlines(y=med, xmin=start_ind, xmax=start_ind+values[i], linewidth=2, color=mycolors[i])
                start_ind += values[i]

            # Plot stacked bar plot
            lefts = 0
            for i in range(len(values)):
                v = values[i]
                if i == 0:
                    ax1.barh(10, v, height=10, label=f'{keys[i]}', color=mycolors[i])
                    lefts = v
                else:
                    ax1.barh(10, v, left=lefts, height=10, label=f'{keys[i]}', color=mycolors[i])
                    lefts += v

            my_ncol = 4
            my_yheight = 20
            if question["prefix"] == "dailyaction":
                my_ncol = 3
                my_yheight = 30
            leg = ax1.legend(loc='upper left', ncol=my_ncol, mode="expand", title="$\\bf{Median}:$")
            leg._legend_box.align = "left"
            plt.ylabel('Percentage')
            plt.xlabel('Stimulus')
            yb, yt = ax1.get_ylim()
            ax1.set_ylim(yb, yt + my_yheight)
            plt.savefig(f'{save_dir}/{question["prefix"]}_consensus_max_category_{suffix}',
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


