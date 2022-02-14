from behavioral_analysis import BehavioralAnalysis
from questions import get_questions
from calculate_statistics import calculate_statistics


ba = BehavioralAnalysis(ba_number='2')

questions = get_questions()
calculate_statistics(questions)



for q in questions:
    if 'categories' in q:
        ba.barplot(q)
    else:
        pass
        #ba.create_boxplots(desc=q, stats=s, only_hist=True)

