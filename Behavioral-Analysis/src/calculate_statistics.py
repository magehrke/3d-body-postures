import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import utils


def calculate_statistics(questions: [dict], force: bool = False) -> None:
    print(f'--- CALCULATING STATISTICS ---')
    # --- BEHAVIORAL ANALYSIS 1 ---

    uparam_dict_1 = utils.get_uparams_ba_1()
    uparam_dict_2 = utils.get_uparams_ba_2()

    calculate_stats_per_question(questions, uparam_dict_1, uparam_dict_2, force)


def calculate_stats_per_question(questions: [dict], uparam_dict_1, uparam_dict_2, force) -> None:
    """
    Calculate statistics of the behavioral analysis(mean, median, std, n and raw data values)
    for a question for each pose over all participants. Viewpoints of the same pose are stored
    together under their respective 'uparam' name.

    We differentiate between a likert and a categorical question type, where the latter
    had several answers to choose from.

    :param questions: dictionary containing the information of a question
    :param uparam_dict_1: dictionary storing the raw answers for each stimulus/uparam for analysis 1
    :param uparam_dict_2: dictionary storing the raw answers for each stimulus/uparam for analysis 2
    :param force: if true, statistics are recalculated, even if they are already available on disk
    """
    if not os.path.exists(f'../output/behavioral_analysis_1/stat_dicts/'):
        os.makedirs(f'../output/behavioral_analysis_1/stat_dicts/')
    if not os.path.exists(f'../output/behavioral_analysis_2/stat_dicts/'):
        os.makedirs(f'../output/behavioral_analysis_2/stat_dicts/')

    for quest_dict in questions:
        for ba_num in quest_dict['behavioral_analysis']:
            out_dir = f'../output/behavioral_analysis_{ba_num}/'
            save_dir = f'{out_dir}stat_dicts/{quest_dict["prefix"]}_dict.pkl'

            if not os.path.exists(save_dir) or force:
                if ba_num == 1:
                    uparam_dict = uparam_dict_1
                else:
                    uparam_dict = uparam_dict_2
                stats = {}
                loop_desc = f'Generating {quest_dict["prefix"]} statistics for BA {ba_num}'
                for uparam_name, dfs_of_vps in tqdm(uparam_dict.items(), loop_desc):
                    uparam_stats = {}
                    for i, (pose_name, df) in enumerate(dfs_of_vps.items()):
                        questions = df.iloc[0, :]
                        hit = questions.str.contains(quest_dict['question'])
                        assert hit.sum() == 1
                        raw = df.loc[2:, hit]
                        raw = raw.dropna()
                        if 'categories' in quest_dict:  # "Categorical" question
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
                                    num = num[num.rindex(',') + 1:]
                                    raw.append(int(num))
                            raw = np.array(raw)
                            desc = {'raw': raw}
                            uparam_stats[pose_name] = desc
                        else:  # Likert question
                            raw = raw.apply(pd.to_numeric)
                            raw = raw.to_numpy().flatten()
                            desc = {'mean': np.mean(raw), 'std': np.std(raw),
                                    'median': np.median(raw), 'raw': raw, 'n': len(raw)}
                            uparam_stats[pose_name] = desc
                    stats[uparam_name] = uparam_stats
                with open(save_dir, "wb") as output_file:
                    pickle.dump(stats, output_file)

    # --- CALCULATE STATS FOR WITH ALL (BOTH) Behavioral Analysis Together --- #
    if not os.path.exists(f'../output/all/stat_dicts/'):
        os.makedirs(f'../output/all/stat_dicts/')

    q_names = ['emotion', 'bodypart', 'realism', 'dailyaction', 'possibility', 'movement']
    for q_name in q_names:
        save_dir = f'../output/all/stat_dicts/{q_name}_dict.pkl'
        if not os.path.exists(save_dir) or force:
            stats = {}
            with open(f'../output/behavioral_analysis_1/stat_dicts/{q_name}_dict.pkl', "rb") as input_file:
                stats_1 = pickle.load(input_file)
            with open(f'../output/behavioral_analysis_2/stat_dicts/{q_name}_dict.pkl', "rb") as input_file:
                stats_2 = pickle.load(input_file)
            for uparam_name in stats_1:
                uparam_stats = {}
                uparam_1 = stats_1[uparam_name]
                uparam_2 = stats_2[uparam_name]
                for pose_name in uparam_1:
                    raw_1 = uparam_1[pose_name]['raw']
                    raw_2 = uparam_2[pose_name]['raw']
                    # Flip likert scale for possibility question in the first analysis
                    if q_name == 'possibility':
                        raw_1 = np.array([6 - x for x in raw_1])
                    desc = {'raw': np.concatenate((raw_1, raw_2))}
                    uparam_stats[pose_name] = desc
                stats[uparam_name] = uparam_stats
            with open(save_dir, "wb") as output_file:
                pickle.dump(stats, output_file)




