from behavioral_analysis import BehavioralAnalysis
from questions import get_questions
from calculate_statistics import calculate_statistics

"""
    File: main.py
    Author: MA Gehrke
    Date: 14.02.2022

    Analyse the data of the bahavioral analysis of the
    3D body posture questionnaire, including bar- & boxplots.

    Before any analysis can be done, the data needs to be down-
    loaded from qualtrics. Go to Data & Analyses -> Export & Import 
    -> Export Data... -> CSV, tick 'Download all fields' and 'Use
    numeric values', and press Download.
"""

ba = BehavioralAnalysis(ba_numbers=['2', '1', 'all'])

questions = get_questions()
calculate_statistics(questions, force=False)

for q in questions:
    # Barplots only for questions that had categories to choose from
    if 'categories' in q:
        ba.barplot(question=q, hist_only=False)
    # all others are likert-scale questions where we can create boxplots
    else:
        ba.create_boxplots(question=q, only_hist=False)

