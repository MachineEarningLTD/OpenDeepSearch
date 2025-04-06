import pandas as pd

dataframe_paths = [
    'out/Fabian_1_to_200.jsonl',
    'out/nicolas.jsonl',
    'out/nicolas_2.jsonl',
    'out/dennis_run1.jsonl',
    'out/dennis_run2.jsonl',
    'out/lionel.jsonl',
]
dataframes = []
for path in dataframe_paths:
    with open(path, 'r') as f:
        data = pd.read_json(path, lines=True)
        dataframes.append(data)

final_file = 'out/dataframe.jsonl'

dataframes[3] = dataframes[3].iloc[:40]


final_frame = pd.concat(dataframes, ignore_index=True)

for idx, row in final_frame.iterrows():
    final_frame.at[idx, 'idx'] = idx

final_frame.to_json(final_file, orient='records', lines=True)