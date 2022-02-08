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

    Analyse the data of the bahavioral analysis of the
    3D body posture questionnaire, including bar- & boxplots.  
"""


class BehavioralAnalysis:
    def __init__(self):
        # Resolution of the plots
        self.dpi = 200

        df = pd.read_csv(f'../input/behavioral_analysis_1/Questionnaire.csv')
        print(f'Original DF shape: {df.shape}')

        # Drop columns
        drop_names_starting_with = ['Timing', 'Rest time']
        for n in drop_names_starting_with:
            vec = df.iloc[0, :].str.startswith(n)
            df.drop(columns=df.loc[:, vec].columns.values, inplace=True)

        # Drop unfinished observations
        df = df[df['Finished'] != '0']
        print(f'DF shape after dropping: {df.shape}')
        print(f'Number of Subjects: {df.shape[0] - 2}')
        c_names = df.columns.values

        # Load file containing which question belongs to which pose
        p_items_lst = np.load('item-question-association.npy', allow_pickle=True).item()
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

        # Extract only those items that are in the df
        # This should result in 10 questions per item
        # Save the questions in a double dictionary
        # dict[uparam_name][full_name_per_viewpoint]
        self.uparam_dict = {}
        for sn in uparam_names:
            viewpoint_dict = {}
            for pose_name in p_items_lst.keys():
                if pose_name.startswith(f'{sn}_'):
                    lst = []
                    it = p_items_lst[pose_name]
                    for i in it:
                        if i in c_names:
                            lst.append(i)
                        elif f'{i}_1' in c_names:
                            lst.append(f'{i}_1')
                    assert len(lst) == 10
                    viewpoint_dict[pose_name] = df[lst]
            self.uparam_dict[sn] = viewpoint_dict

    def get_statistics(self, quest_dict: dict):
        save_dir = f'output/stat_dics/{quest_dict["prefix"]}_dict.pkl'
        if os.path.exists(save_dir):
            with open(save_dir, "rb") as input_file:
                stats = pickle.load(input_file)
        else:
            stats = {}
            loop_desc = f'Generating {quest_dict["prefix"]} statistics'
            for uparam_name, dfs_of_vps in tqdm(self.uparam_dict.items(), loop_desc):
                uparam_stats = {}
                for i, (pose_name, df) in enumerate(dfs_of_vps.items()):
                    questions = df.iloc[0, :]
                    hit = questions.str.contains(quest_dict['question'])
                    assert hit.sum() == 1
                    raw = df.loc[2:, hit]
                    raw = raw.dropna()
                    raw = raw.apply(pd.to_numeric)
                    raw = raw.to_numpy().flatten()
                    desc = {'mean': np.mean(raw), 'std': np.std(raw),
                            'median': np.median(raw), 'raw': raw, 'n': len(raw)}
                    uparam_stats[pose_name] = desc
                stats[uparam_name] = uparam_stats
            with open(save_dir, "wb") as output_file:
                pickle.dump(stats, output_file)
        return stats

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

    def create_boxplots(self, desc: dict, stats: dict, only_hist=False):
        loop_desc = f'Creating {desc["prefix"]} boxplots'
        save_dir = f'boxplots/{desc["prefix"]}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        scatter_data = []
        for uparam, dfs_of_vps in tqdm(stats.items(), loop_desc):
            raw_all = []
            for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                raw_all.extend(vp_dict['raw'])
            scatter_data.append([uparam, np.mean(raw_all)])
            if not only_hist:
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f'{uparam} (Scale = {self.scale[uparam]})')

                ax[0].boxplot(raw_all, positions=[0])
                ax[0].plot([0], np.mean(raw_all), '+', label='Mean')
                ax[0].set_ylim((desc['likert_min'] - 1, desc['likert_max'] + 1))
                ax[0].set_yticks(range(desc['likert_min'], desc['likert_max'] + 1))
                ax[0].set_yticklabels(range(desc['likert_min'],
                                            desc['likert_max'] + 1))
                ax[0].set_ylabel(f'{desc["likert_str_min"]}     =>     '
                                 f'{desc["likert_str_max"]}')
                ax[0].tick_params(labelbottom=False, bottom=False)
                ax[0].legend(frameon=False)

                im = mpimg.imread(f'Stim_images/{uparam}_Viewpoint_2_scale_'
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
        plt.xlabel(f'{desc["likert_str_min"]}     =>     '
                   f'{desc["likert_str_max"]}')
        plt.xlim((desc['likert_min'] - 1, desc['likert_max'] + 1))
        plt.xticks(range(desc['likert_min'], desc['likert_max'] + 1),
                   range(desc['likert_min'], desc['likert_max'] + 1))
        mu, std = norm.fit(float_value_arr)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2,
                 label=f'mu = {round(mu, 2)}, std = {round(std, 2)}')
        plt.legend()
        plt.savefig(f'{save_dir}/{desc["prefix"]}_hist',
                    dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # Print & export overview of values
        out_file = open(f'{save_dir}/{desc["prefix"]}_values.txt', 'w')
        out_file.write(f'Stimuli, {desc["likert_str_min"]} '
                       f'({desc["likert_min"]}) => {desc["likert_str_max"]}'
                       f'({desc["likert_max"]})\n\n')
        for i in range(len(scatter_data[0])):
            out_file.write(f'{format(scatter_data[0][i] + ",", " <22")}'
                           f'{round(float(scatter_data[1][i]), 2)}\n')
            if i % 10 == 9:
                out_file.write('\n')
        out_file.close()

    def get_statistics_cat(self, quest_dict: dict):
        save_dir = f'output/stat_dics/{quest_dict["prefix"]}_dict.pkl'
        if os.path.exists(save_dir):
            with open(save_dir, "rb") as input_file:
                stats = pickle.load(input_file)
        else:
            stats = {}
            loop_desc = f'Generating {quest_dict["prefix"]} statistics'
            for uparam_name, dfs_of_vps in tqdm(self.uparam_dict.items(), loop_desc):
                uparam_stats = {}
                for i, (pose_name, df) in enumerate(dfs_of_vps.items()):
                    questions = df.iloc[0, :]
                    hit = questions.str.contains(quest_dict['question'])
                    assert hit.sum() == 1
                    raw = df.loc[2:, hit]
                    raw = raw.dropna()
                    _raw = raw.to_numpy().flatten()
                    # If people clicked yes and chose an emotion
                    # sometimes it is saved as '1,7'. So we have to
                    # extract the emotion.
                    raw = []
                    for num in _raw:
                        try:
                            raw.append(int(num))
                        except ValueError:
                            assert type(num) == str
                            num = num[num.rindex(',')+1:]  # TODO: last num?
                            raw.append(int(num))
                    raw = np.array(raw)
                    desc = {
                        'raw': raw
                    }
                    uparam_stats[pose_name] = desc
                stats[uparam_name] = uparam_stats
            with open(save_dir, "wb") as output_file:
                pickle.dump(stats, output_file)
        return stats

    def barplot_cat(self, desc: dict, stats, hist_only=False):
        loop_desc = f'Creating {desc["prefix"]} barplots'
        save_dir = f'barplots/{desc["prefix"]}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        hist_data = []
        for uparam, dfs_of_vps in tqdm(stats.items(), loop_desc):
            raw_all = []
            for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                raw_all.extend(vp_dict['raw'])
            count = Counter(raw_all)
            names = list(desc['categories'].values())
            keys = list(desc['categories'].keys())
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

                im = mpimg.imread(f'Stim_images/{uparam}_Viewpoint_2_scale_'
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
        plt.savefig(f'{save_dir}/{desc["prefix"]}_bar',
                    dpi=self.dpi, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":

    ba = BehavioralAnalysis()

    # ===== HIGH LEVEL FEATURES - EMOTION ====== #

    # EMOTION
    # Question type: yes/no
    # Possibilities if yes: Sadness, Happiness, Fear, Disgust, Anger, Surprise
    emo_desc = {
        'question': f'Does the posture show any emotion?',
        'categories': {  # From Questionaire -> question -> recode values
            9: 'Yes', 8: 'No', 16: 'Sadness', 17: 'Happiness', 18: 'Fear',
            19: 'Disgust', 20: 'Anger', 21: 'Surprise'
        },
        'prefix': 'emotion'
    }
    emo_stats = ba.get_statistics_cat(emo_desc)

    # BODY PART
    body_desc = {
        'question': f'Which body part did you mostly look at?',
        'categories': {
            1: 'Head', 2: 'Hands', 3: 'Arms', 4: 'Legs', 5: 'Feet',
            6: 'Overall'
        },
        'prefix': 'bodypart'
    }
    body_stats = ba.get_statistics_cat(body_desc)

    # AROUSAL
    # (Boring, 2, 3, 4, Arousing)
    arou_desc = {
        'question': f'Do you feel this posture arousing or rather boring?',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Boring',
        'likert_str_max': 'Arousing', 'prefix': 'arousal'
    }
    arou_stats = ba.get_statistics(arou_desc)

    # POSITIVITY
    # (Very negative, 2, 3, 4, Very positive)
    pos_desc = {
        'question': f'Do you feel this posture is\npositive or rather negative?',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Very negative',
        'likert_str_max': 'Very positive', 'prefix': 'positivity'
    }
    pos_stats = ba.get_statistics(pos_desc)

    # ===== HIGH LEVEL FEATURES - ACTION ====== #

    # FAMILIARITY
    # (Very unfamiliar, 2, 3, 4, Very familiar)
    fam_desc = {
        'question': f'Is this posture familiar to you?', 'likert_min': 1,
        'likert_max': 5, 'likert_str_min': 'Very unfamiliar',
        'likert_str_max': 'Very familiar', 'prefix': 'familiarity'
    }
    fam_stats = ba.get_statistics(fam_desc)

    # REALISM
    # (Very unrealistic, 2, 3, 4, Very realistic)
    real_desc = {
        'question': f'Is this a realistic body posture you can make yourself',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Very unrealistic',
        'likert_str_max': 'Very realistic', 'prefix': 'realism'
    }
    real_stats = ba.get_statistics(real_desc)

    # DAILY ACTION
    # Question type: yes/no + choice if yes
    daily_desc = {
        'question': f'Can you recognize a daily action in the posture?',
        'categories': {  # From Questionaire -> question -> recode values
            1: 'Yes', 2: 'No', 3: 'Greeting a person', 4: 'Grasping an object',
            5: 'Catching an object', 6: 'Self-Defending', 7: 'None of the above'
        },
        'prefix': 'dailyaction'
    }
    daily_stats = ba.get_statistics_cat(daily_desc)

    # POSSIBILITY
    # (Possible, 2, 3, 4, Impossible)
    poss_desc = {
        'question': f'Is it possible for any of the body parts to be in this',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Possible', 'likert_str_max': 'Impossible',
        'prefix': 'possibility'
    }
    poss_stats = ba.get_statistics(poss_desc)

    # ===== MID LEVEL FEATURES - MOVEMENT CHARACTERISTICS ====== #

    # MOVEMENT
    # (Little movement, 2, 3, 4, A lot of movement)
    mov_desc = {
        'question': f'How much overall body movement is implied in the posture?',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Little movement', 'likert_str_max': 'A lot of movement',
        'prefix': 'movement'
    }
    mov_stats = ba.get_statistics(mov_desc)

    # CONTRACTION
    # (Little contraction, 2, 3, 4, A lot of contraction)
    cont_desc = {
        'question': f'How much body contraction is there in the body posture?',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Little contraction', 'likert_str_max': 'A lot of contraction',
        'prefix': 'contraction'
    }
    cont_stats = ba.get_statistics(cont_desc)

    # COMPUTE STUFF
    #ba.barplot_cat(desc=daily_desc, stats=daily_stats, hist_only=False)
    #ba.barplot_cat(desc=emo_desc, stats=emo_stats, hist_only=False)
    #ba.barplot_cat(desc=body_desc, stats=body_stats, hist_only=False)
    ba.create_boxplots(desc=arou_desc, stats=arou_stats, only_hist=True)
    ba.create_boxplots(desc=pos_desc, stats=pos_stats, only_hist=True)
    ba.create_boxplots(desc=fam_desc, stats=fam_stats, only_hist=True)
    ba.create_boxplots(desc=real_desc, stats=real_stats, only_hist=True)
    ba.create_boxplots(desc=poss_desc, stats=poss_stats, only_hist=True)
    ba.create_boxplots(desc=mov_desc, stats=mov_stats, only_hist=True)
    ba.create_boxplots(desc=cont_desc, stats=cont_stats, only_hist=True)

