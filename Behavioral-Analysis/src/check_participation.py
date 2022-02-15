import pandas as pd

df = pd.read_csv(f'../input/behavioral_analysis_2/Questionnaire.csv')
print(f'Number of Subjects: {df.shape[0] - 2}')


pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 500)


df = df[2:]
print(df[['Status', 'Progress', 'RecordedDate', 'Duration (in seconds)', 'Finished', 'Q13']])
