import os
import pandas as pd

#Place the files that are to be merged in current working directory

cwd = os.path.abspath('')
print(cwd)
files = os.listdir(cwd)
df = pd.DataFrame()
for file in files:
    if file.endswith('.csv'):
        df = df.append(pd.read_csv(file), ignore_index=True)
df.head()
df.to_csv('Complete_Data.csv')
