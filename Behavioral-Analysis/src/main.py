from behavioral_analysis import BehavioralAnalysis
from questions import get_questions
from calculate_statistics import calculate_statistics

"""
    File: main.py
    Author: MA Gehrke
    Date: 14.02.2022

    Analyse the data of the bahavioral analysis of the
    3D body posture questionnaire, including bar- & boxplots.

    BEFORE THE ANALYSIS
    -------------------
    Before any analysis can be done, the data needs to be down-
    loaded from qualtrics. Go to Data & Analyses -> Export & Import 
    -> Export Data... -> CSV, tick 'Download all fields' and 'Use
    numeric values', and press Download.
    
    To associate the questions with the images that have been shown
    on the screen, the file "associate_items_poses.py" has to be
    executed once before running any analysis.
"""

ba = BehavioralAnalysis(ba_numbers=['all'])

# Get all questions that have been used in any behavioral analyses
questions = get_questions()

"""
    Calculate the statistics for each question and save
    it into a dictionary. If a dictionary already exists,
    it will not be calculated again, except when the 
    'force' option is set to 'true'. 
"""
calculate_statistics(questions, force=False)

for q in questions:
    # Barplots only for questions that had categories to choose from
    if 'categories' in q:
        ba.barplot(question=q, sum_viewpoints=False, hist_only=True)
        xxx = 'placeholder'
    # all others are likert-scale questions where we can create boxplots
    else:
        #ba.create_boxplots_histograms(question=q, sum_viewpoints=False, only_hist=False)
        xxx = 'placeholder'
