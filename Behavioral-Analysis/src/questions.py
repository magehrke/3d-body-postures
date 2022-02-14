
"""
    File: questions.py
    Author: MA Gehrke
    Date: 11.02.2022

    This file contains the questions (wrapped in
    dictionaries) that we used during the first
    and second behavioral analysis.
"""


def get_questions():
    """
    We have two different kind of questions: questions
    with a likert scale and questions with predefined
    categories.
    """
    questions = []

    # ===== HIGH LEVEL FEATURES - EMOTION ====== #

    # EMOTION
    # Type: yes/no - if yes: Sadness, Happiness, Fear, Disgust, Anger, Surprise
    questions.append({
        'question': f'Does the posture show any emotion?',
        'categories': {  # From Questionaire -> question -> recode values
            9: 'Yes', 8: 'No', 16: 'Sadness', 17: 'Happiness', 18: 'Fear',
            19: 'Disgust', 20: 'Anger', 21: 'Surprise'
        }, 'prefix': 'emotion', 'behavioral_analysis': [1, 2]
    })

    # BODY PART
    questions.append({
        'question': f'Which body part did you mostly look at?',
        'categories': {
            1: 'Head', 2: 'Hands', 3: 'Arms', 4: 'Legs', 5: 'Feet',
            6: 'Overall'
        }, 'prefix': 'bodypart', 'behavioral_analysis': [1, 2]
    })

    # AROUSAL - (Boring, 2, 3, 4, Arousing)
    questions.append({
        'question': f'Do you feel this posture arousing or rather boring?',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Boring',
        'likert_str_max': 'Arousing', 'prefix': 'arousal',
        'behavioral_analysis': [1]
    })

    # POSITIVITY - (Very negative, 2, 3, 4, Very positive)
    questions.append({
        'question': f'Do you feel this posture is\npositive or rather negative?',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Very negative',
        'likert_str_max': 'Very positive', 'prefix': 'positivity',
        'behavioral_analysis': [1]
    })

    # ===== HIGH LEVEL FEATURES - ACTION ====== #

    # FAMILIARITY - (Very unfamiliar, 2, 3, 4, Very familiar)
    questions.append({
        'question': f'Is this posture familiar to you?', 'likert_min': 1,
        'likert_max': 5, 'likert_str_min': 'Very unfamiliar',
        'likert_str_max': 'Very familiar', 'prefix': 'familiarity',
        'behavioral_analysis': [1]
    })

    # REALISM - (Very unrealistic, 2, 3, 4, Very realistic)
    questions.append({
        'question': f'Is this a realistic body posture you can make yourself',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Very unrealistic',
        'likert_str_max': 'Very realistic', 'prefix': 'realism',
        'behavioral_analysis': [1]
    })
    questions.append({
        'question': f'Is this a realistic posture that you can adopt for yourself',
        'likert_min': 1, 'likert_max': 5, 'likert_str_min': 'Very unrealistic',
        'likert_str_max': 'Very realistic', 'prefix': 'realism',
        'behavioral_analysis': [2]
    })

    # DAILY ACTION - Question type: yes/no + choice if yes
    questions.append({
        'question': f'Can you recognize a daily action in the posture?',
        'categories': {  # From Questionaire -> question -> recode values
            1: 'Yes', 2: 'No', 3: 'Greeting a person', 4: 'Grasping an object',
            5: 'Catching an object', 6: 'Self-Defending', 7: 'None of the above'
        }, 'prefix': 'dailyaction', 'behavioral_analysis': [1, 2]
    })

    # POSSIBILITY - (Possible, 2, 3, 4, Impossible)
    questions.append({
        'question': f'Is it possible for any of the body parts to be in this',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Possible', 'likert_str_max': 'Impossible',
        'prefix': 'possibility', 'behavioral_analysis': [1]
    })
    questions.append({
        'question': f'Is it possible for all body parts to be in their respective',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Impossible', 'likert_str_max': 'Possible',
        'prefix': 'possibility', 'behavioral_analysis': [2]
    })

    # ===== MID LEVEL FEATURES - MOVEMENT CHARACTERISTICS ====== #

    # MOVEMENT - (Little movement, 2, 3, 4, A lot of movement)
    questions.append({
        'question': f'How much overall body movement is implied in the posture?',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Little movement', 'likert_str_max': 'A lot of movement',
        'prefix': 'movement', 'behavioral_analysis': [1]
    })
    questions.append({
        'question': f'How much movement do you see in the image',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'No movement', 'likert_str_max': 'A lot of movement',
        'prefix': 'movement', 'behavioral_analysis': [2]
    })

    # CONTRACTION - (Little contraction, 2, 3, 4, A lot of contraction)
    questions.append({
        'question': f'How much body contraction is there in the body posture?',
        'likert_min': 1, 'likert_max': 5,
        'likert_str_min': 'Little contraction', 'likert_str_max': 'A lot of contraction',
        'prefix': 'contraction',
        'behavioral_analysis': [1]
    })

    return questions
