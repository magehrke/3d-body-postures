import pandas as pd
import numpy as np


def get_data_and_drop_unfinished(ba_num: int, print_info: bool = False) -> pd.DataFrame:
    """
    Read the data from questionnaire csv file into
    pandas DataFrame and drop unfinished observations
    and columns regarding time.

    :param ba_num: number of the behavioral analysis we want the data from
    :param print_info: printing infos, e.g. original shape and subjects

    :return: The pandas DataFrame containing the data from the csv.
    """
    assert ba_num in [1, 2]
    # Define paths
    paths = {
        1: f'../input/behavioral_analysis_1/Questionnaire.csv',
        2: f'../input/behavioral_analysis_2/Questionnaire.csv'
    }

    # Get data
    df = pd.read_csv(paths[ba_num])
    if print_info:
        print(f'BA{ba_num} Original DF shape: {df.shape}')

    # Drop columns
    drop_names_starting_with = ['Timing', 'Rest time']
    for n in drop_names_starting_with:
        vec = df.iloc[0, :].str.contains(n)
        df.drop(columns=df.loc[:, vec].columns.values, inplace=True)

    # Drop unfinished observations
    df = df[df['Finished'] != '0']
    if print_info:
        print(f'BA{ba_num} DF shape after dropping: {df.shape}')
        print(f'BA{ba_num} Number of Subjects: {df.shape[0] - 2}')

    return df


def get_uparams_ba_1():
    """
    Load the data for behavioral analysis 1, drop unfinished
    questionnaires and extract the answers to each question for
    each uparam (stimulus).

    Returns
    -------
    uparam_dict : dict
        A dictionary with dictionaries. Outer keys are the uparams
        without viewpoint. Inner keys are uparams with viewpoints.
        Values are all the answers to each uparam (stimulus).
    """

    df = get_data_and_drop_unfinished(1, True)
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
    """
    Load the data for behavioral analysis 2, drop unfinished
    questionnaires and extract the answers to each question for
    each uparam (stimulus).

    Returns
    -------
    uparam_dict : dict
        A dictionary with dictionaries. Outer keys are the uparams
        without viewpoint. Inner keys are uparams with viewpoints.
        Values are all the answers to each uparam (stimulus).
    """
    df = get_data_and_drop_unfinished(2, True)
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

    '''
    Extract only those items that are in the df
    This should result in 6 questions per item
    Save the questions in a double dictionary
    dict[uparam_name][full_name_per_viewpoint]
    '''
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
