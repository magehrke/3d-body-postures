import utils
from tqdm import tqdm
from collections import Counter
import numpy as np
from datetime import datetime

"""
    File: demographics.py
    Author: MA Gehrke
    Date: 05.07.2022

    Calculate the statics of the behavioral 
    analysis and print it to the console.
"""

quest_id_ba1 = {
    'nationality': 'QID9',
    'handedness': 'QID111',
    'birthdate': 'QID3',
    'gender': 'QID4',
    'ethnicity': 'QID5',
    'ethnicity_text': 'QID5_4_TEXT',
    'impairments': 'QID10'
}

quest_id_ba2 = {
    'nationality': 'Q7',
    'handedness': 'Q8',
    'birthdate': 'Q9',
    'gender': 'Q10',
    'ethnicity': 'Q11',
    'ethnicity_text': 'Q11_4_TEXT',
    'impairments': 'Q12'
}

answer_options = {
    'handedness': {
        1: 'right',
        2: 'left'
    },
    'birthdate': '(dd/mm/yyyy)',
    'gender': {
        1: 'male',
        2: 'female',
        3: 'non-binary / third gender',
        4: 'prefer not to say'
    },
    'ethnicity': {
        1: 'white',
        2: 'black',
        3: 'asian',
        4: 'other (-> look below)'
    }
}


def calculate_demographics() -> None:
    """
    Calculate and print the demographics of the behavioral
    analysis to the console.
    """

    df_ba1 = utils.get_data_and_drop_unfinished(1)
    df_ba2 = utils.get_data_and_drop_unfinished(2)

    # Nationality
    nat1 = df_ba1[quest_id_ba1['nationality']].loc[2:].to_numpy()
    nat2 = df_ba2[quest_id_ba2['nationality']].loc[2:].to_numpy()
    nat = np.append(nat1, nat2, axis=0)
    nat = np.array([x.strip().lower() for x in nat])
    counter = Counter(nat)
    counter = sorted(counter.items(), key=lambda pair: pair[1])
    print('\nNationality\n----------')
    print(f'Total: {nat.shape[0]}')
    print(counter)

    # Handedness
    hand1 = df_ba1[quest_id_ba1['handedness']].loc[2:].dropna().to_numpy()
    hand2 = df_ba2[quest_id_ba2['handedness']].loc[2:].dropna().to_numpy()
    hand = np.append(hand1, hand2, axis=0)
    hand = np.array([answer_options['handedness'][int(x)] for x in hand])
    print('\nHandedness\n----------')
    print(f'Total: {hand.shape[0]}')
    print(Counter(hand))

    # Birthdate
    birth1 = df_ba1[quest_id_ba1['birthdate']].loc[2:].to_numpy()
    birth2 = df_ba2[quest_id_ba2['birthdate']].loc[2:].to_numpy()
    birth = np.append(birth1, birth2, axis=0)
    end1 = df_ba1['EndDate'].loc[2:].to_numpy()
    end2 = df_ba2['EndDate'].loc[2:].to_numpy()
    end = np.append(end1, end2, axis=0)
    age = []
    for i in range(birth.shape[0]):
        date_start = birth[i].replace('.', '/')
        date_start = date_start.replace('-', '/')
        date_start = datetime.strptime(date_start, '%m/%d/%Y')
        date_end = datetime.strptime(end[i], '%Y-%m-%d %H:%M:%S')
        date = date_end - date_start
        age_i = np.floor(np.array(date.days) / 365)
        if age_i > 0:  # one participant put in the current year, not birth year
            age.append(age_i)
    age = np.array(age, dtype=int)
    print('\nAge\n-----------')
    print(f'Total: {age.shape[0]}')
    print(f'Mean = {np.mean(age)}, Std = {np.std(age)}')
    print(f'Max = {np.max(age)}, Min = {np.min(age)}')

    # Gender
    gen1 = df_ba1[quest_id_ba1['gender']].loc[2:].dropna().to_numpy()
    gen2 = df_ba2[quest_id_ba2['gender']].loc[2:].dropna().to_numpy()
    gen = np.append(gen1, gen2, axis=0)
    gen = np.array([answer_options['gender'][int(x)] for x in gen])
    print('\nGender\n----------')
    print(f'Total: {gen.shape[0]}')
    print(Counter(gen))

    # Ethnicity
    eth1 = df_ba1[quest_id_ba1['ethnicity']].loc[2:].to_numpy()
    eth2 = df_ba2[quest_id_ba2['ethnicity']].loc[2:].to_numpy()
    eth = np.append(eth1, eth2, axis=0)
    eth = np.array([answer_options['ethnicity'][int(x)] for x in eth])
    print('\nEthnicity\n----------')
    print(f'Total: {eth.shape[0]}')
    print(Counter(eth))
    eth_txt1 = df_ba1[quest_id_ba1['ethnicity_text']].loc[2:].dropna().to_numpy()
    eth_txt2 = df_ba2[quest_id_ba2['ethnicity_text']].loc[2:].dropna().to_numpy()
    eth_txt = np.append(eth_txt1, eth_txt2, axis=0)
    print(f'Other: {eth_txt}')

    # Impairments
    imp1 = df_ba1[quest_id_ba1['impairments']].loc[2:].to_numpy()
    imp2 = df_ba2[quest_id_ba2['impairments']].loc[2:].to_numpy()
    imp = np.append(imp1, imp2, axis=0)
    no_words = ['no', '-', 'none', 'no ', 'No.', '/']
    imp = np.array(['No' if x in no_words else x for x in imp])
    print('\nImpairments\n----------')
    print(f'Total: {imp.shape[0]}')
    print(Counter(imp))


if __name__ == "__main__":
    calculate_demographics()

