import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from collections import Counter


"""
    QUESTIONS
    - Does scale has an effect?
    - Does viewpoint has an effect?
"""

class BehavioralAnalysis:
    def __init__(self):
        self.dpi = 50

        df = pd.read_csv(f'Questionnaire.csv')
        print(f'Original DF shape: {df.shape}')

        # Drop columns
        drop_names_starting_with = ['Timing', 'Rest time']
        for n in drop_names_starting_with:
            vec = df.iloc[0, :].str.startswith(n)
            df.drop(columns=df.loc[:, vec].columns.values, inplace=True)

        # Drop unfinished observations
        df = df[df['Finished'] != '0']
        print(f'DF shape after dropping: {df.shape}')
        c_names = df.columns.values

        # Load file containing which question belongs to which pose
        p_items_lst = np.load('items-questions.npy', allow_pickle=True).item()
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
                if stim.startswith(un):
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
        save_dir = f'stat_dics/{quest_dict["prefix"]}_dict.pkl'
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
                    raw = raw.apply(pd.to_numeric)
                    raw = raw.replace([quest_dict['likert_str_min'], quest_dict['likert_min']])
                    raw = raw.replace([quest_dict['likert_str_max'], quest_dict['likert_max']]).to_numpy()
                    raw = raw[~pd.isnull(raw)]
                    raw = raw.astype(np.int)
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

    def create_boxplots(self, desc: dict, stats: dict):
        loop_desc = f'Creating {desc["prefix"]} boxplots'
        save_dir = f'boxplots/{desc["prefix"]}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for uparam, dfs_of_vps in tqdm(stats.items(), loop_desc):
            raw_all = []
            for i, (pose_name, vp_dict) in enumerate(dfs_of_vps.items()):
                raw_all.extend(vp_dict['raw'])

            plt.boxplot(raw_all)
            plt.title(f'{uparam} (Scale = {self.scale[uparam]})')
            plt.ylim((desc['likert_min'] - 1, desc['likert_max'] + 1))
            plt.yticks(range(desc['likert_min'], desc['likert_max'] + 1),
                       range(desc['likert_min'],  desc['likert_max'] + 1))
            plt.ylabel('Likert Scale')
            plt.savefig(f'{save_dir}/{uparam}_scale_{self.scale[uparam]}',
                        dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def get_statistics_cat(self, quest_dict: dict):
        save_dir = f'stat_dics/{quest_dict["prefix"]}_dict.pkl'
        if not os.path.exists(save_dir):
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
                    raw = np.array(raw.iloc[:, 0].values.tolist())
                    print(raw)
                    raw = raw.astype(np.int)
                    print(raw)
                    for s, t in quest_dict['categories'].items():
                        raw = np.where(raw == s, t, raw)
                    raw = Counter(raw)
                    print(raw)
                    desc = {}
                    uparam_stats[pose_name] = desc
                stats[uparam_name] = uparam_stats
            with open(save_dir, "wb") as output_file:
                pickle.dump(stats, output_file)
        return stats



if __name__ == "__main__":

    ba = BehavioralAnalysis()

    # ===== HIGH LEVEL FEATURES - EMOTION ====== #

    # DAILY ACTION
    # Question type: yes/no + choice if yes
    daily_desc = {
        'question': f'Can you recognize a daily action in the posture?',
        'categories': {  # From Questionaire -> question -> recode values
            1: 'Yes', 2: 'No', 3: 'Greeting a person', 4: 'Grasping an object',
            5: 'Catching an object', 6: 'Self-Defending', 7: 'None of the above'
        },
        'prefix': 'daily'
    }
    #daily_stats = ba.get_statistics_cat(daily_desc)
    #ba.create_boxplots(desc=daily_desc, stats=daily_stats)
    # EMOTION
    # Question type: yes/no
    # Possibilities if yes: Sadness, Happiness, Fear, Disgust, Anger, Surprise
    emo_desc = {
        'question': f'Does the posture show any emotion?',
        'categories': {  # From Questionaire -> question -> recode values
            9: 'Yes', 8: 'No', 16: 'Sadness', 17: 'Happiness', 18: 'Fear',
            19: 'Disgust', 20: 'Anger', 21: 'Surprise'
        },
        'prefix': 'emo'
    }
    #emo_stats = ba.get_statistics_cat(emo_desc)
    #ba.create_boxplots(desc=emo_desc, stats=emo_stats)


    # AROUSAL
    # (Boring, 2, 3, 4, Arousing)
    arou_desc = {
        'question': f'Do you feel this posture arousing or rather boring?',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Boring',
        'likert_str_max': 'Arousing', 'prefix': 'arou'
    }
    arou_stats = ba.get_statistics(arou_desc)
    #ba.create_boxplots(desc=arou_desc, stats=arou_stats)

    # POSITIVITY
    # (Very negative, 2, 3, 4, Very positive)
    pos_desc = {
        'question': f'Do you feel this posture is\npositive or rather negative?',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Very negative',
        'likert_str_max': 'Very positive', 'prefix': 'pos'
    }
    pos_stats = ba.get_statistics(pos_desc)
    #ba.create_boxplots(desc=pos_desc, stats=pos_stats)

    # ===== HIGH LEVEL FEATURES - ACTION ====== #

    # FAMILIARITY
    # (Very unfamiliar, 2, 3, 4, Very familiar)
    fam_desc = {
        'question': f'Is this posture familiar to you?', 'likert_min': 1,
        'likert_max': 5, 'likert_str_min': 'Very unfamiliar',
        'likert_str_max': 'Very familiar', 'prefix': 'fam'
    }
    fam_stats = ba.get_statistics(fam_desc)
    ba.create_boxplots(desc=fam_desc, stats=fam_stats)

    # REALISM
    # (Very unrealistic, 2, 3, 4, Very realistic)
    real_desc = {
        'question': f'Is this a realistic body posture you can make yourself',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Very unrealistic',
        'likert_str_max': 'Very realistic', 'prefix': 'real'
    }
    real_stats = ba.get_statistics(real_desc)
    # ba.create_boxplots(desc=real_desc, stats=real_stats)

    # POSSIBILITY
    # (Possible, 2, 3, 4, Impossible)
    poss_desc = {
        'question': f'Is it possible for any of the body parts to be in this',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Possible', 'likert_str_max': 'Impossible',
        'prefix': 'poss'
    }
    poss_stats = ba.get_statistics(poss_desc)
    # ba.create_boxplots(desc=poss_desc, stats=poss_stats)

    # ===== MID LEVEL FEATURES - MOVEMENT CHARACTERISTICS ====== #

    # MOVEMENT
    # (Little movement, 2, 3, 4, A lot of movement)
    mov_desc = {
        'question': f'How much overall body movement is implied in the posture?',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Little movement', 'likert_str_max': 'A lot of movement',
        'prefix': 'mov'
    }
    mov_stats = ba.get_statistics(mov_desc)
    ba.create_boxplots(desc=mov_desc, stats=mov_stats)

    # CONTRACTION
    # (Little contraction, 2, 3, 4, A lot of contraction)
    cont_desc = {
        'question': f'How much body contraction is there in the body posture?',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Little contraction', 'likert_str_max': 'A lot of contraction',
        'prefix': 'cont'
    }
    cont_stats = ba.get_statistics(cont_desc)
    ba.create_boxplots(desc=cont_desc, stats=cont_stats)
