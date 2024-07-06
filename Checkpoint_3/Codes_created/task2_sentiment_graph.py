#K. Anil Kumar
#GEID No: 1011139542

import pandas as pd
import matplotlib.pyplot as plt
#Path to complete data
df = pd.read_csv(r"results\Complete_Data.csv", sep=',', encoding='latin-1')

l = list(df["Company"])
com = input("Enter Company: ")
com = com.lower()
c = 0
for company in l:
    if company in com:
        c = c + 1

ind = l.index(com)
q = []
d = list(df['Sentiment'][ind:ind + c])

for i in range(ind,ind+c):
    q.append(df['Quandrant'][i]+'_'+str(df['Year'][i]))


fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0,0,1,1])

ax.bar(q,d)
plt.show()
