import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm


def calculate_statistics(questions: [dict], force: bool = False) -> None:
    print(f'--- CALCULATING STATISTICS ---')
    # --- BEHAVIORAL ANALYSIS 1 ---

    uparam_dict_1 = get_uparams_ba_1()
    uparam_dict_2 = get_uparams_ba_2()

    calculate_stats_per_question(questions, uparam_dict_1, uparam_dict_2, force)


def get_uparams_ba_1():
    # Get data
    df = pd.read_csv(f'../input/behavioral_analysis_1/Questionnaire.csv')
    print(f'BA1 Original DF shape: {df.shape}')

    # Drop columns
    drop_names_starting_with = ['Timing', 'Rest time']
    for n in drop_names_starting_with:
        vec = df.iloc[0, :].str.contains(n)
        df.drop(columns=df.loc[:, vec].columns.values, inplace=True)

    # Drop unfinished observations
    df = df[df['Finished'] != '0']
    print(f'BA1 DF shape after dropping: {df.shape}')
    print(f'BA1 Number of Subjects: {df.shape[0] - 2}')
    c_names = df.columns.values

    # Load file containing which question belongs to which pose
    p_items_lst = np.load('../input/behavioral_analysis_1/item-question-association.npy', allow_pickle=True).item()
    p_items_lst = dict(p_items_lst)
    assert len(p_items_lst.keys()) == 324

    # Extract uparam names (108 names with 3 viewpoints each)
    uparam_names = [x[:x.find('Viewpoint') - 1] for x in
                    p_items_lst.keys()]
    uparam_names = set(uparam_names)
    assert len(uparam_names) == 108

    # Extract only those items that are in the df
    # This should result in 10 questions per item
    # Create a dictionary that contains the columns
    # of the df that belong to the stimulus.
    # Save the questions in a double dictionary
    # dict[uparam_name][full_name_per_viewpoint]
    uparam_dict = {}
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
        uparam_dict[sn] = viewpoint_dict
    return uparam_dict


def get_uparams_ba_2():
    # Get data
    df = pd.read_csv(f'../input/behavioral_analysis_2/Questionnaire.csv')
    print(f'BA2 Original DF shape: {df.shape}')

    # Drop columns
    drop_names_starting_with = ['Timing', 'Rest time']
    for n in drop_names_starting_with:
        vec = df.iloc[0, :].str.contains(n)
        df.drop(columns=df.loc[:, vec].columns.values, inplace=True)

    # Drop unfinished observations
    df = df[df['Finished'] != '0']
    print(f'BA2 DF shape after dropping: {df.shape}')
    print(f'BA2 Number of Subjects: {df.shape[0] - 2}')
    c_names = df.columns.values

    # Load file containing which question belongs to which pose
    pose_url_df = pd.read_csv('../input/behavioral_analysis_2/im_url_to_pose_stim.csv')
    assert len(pose_url_df) == 324
    assert pose_url_df['pose_name'].nunique() == 324
    assert pose_url_df['url'].nunique() == 324

    # Extract uparam names (108 names with 3 viewpoints each)
    uparam_names = [x[:x.find('Viewpoint') - 1] for x in pose_url_df['pose_name']]
    uparam_names = set(uparam_names)
    assert len(uparam_names) == 108

    # Extract only those items that are in the df
    # This should result in 6 questions per item
    # Save the questions in a double dictionary
    # dict[uparam_name][full_name_per_viewpoint]
    uparam_dict = {}
    for un in uparam_names:
        viewpoint_dict = {}
        for index, row in pose_url_df.iterrows():
            pose_name = row['pose_name']
            im_url = row['url']
            if pose_name.startswith(f'{un}_'):
                sel_cols = []
                # Iterate over the data and find the questions that have
                # been answered after seeing the specific stimuli.
                for i, col in enumerate(df.iloc[0]):
                    if im_url in col:
                        sel_cols.append(c_names[i])
                assert len(sel_cols) == 6
                viewpoint_dict[pose_name] = df[sel_cols]
        uparam_dict[un] = viewpoint_dict
    return uparam_dict


def calculate_stats_per_question(questions: [dict], uparam_dict_1, uparam_dict_2, force) -> dict:
    """
    Load or calculate the statistics (mean, median, std, n or raw data values) for a question
    for each pose over all participants. Viewpoints of the same pose are stored together under
    their respective 'uparam' name.

    We differentiate between a likert and a categorical question type, where the latter
    had several answers to choose from.

    :param questions: dictionary containing the information of a question
    :param force: if this variable is true, we do a new calculation and overwrite old stats on disk
    :return: dictionary with the uparam name as key and a dictionary, containing statistics, as value
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




