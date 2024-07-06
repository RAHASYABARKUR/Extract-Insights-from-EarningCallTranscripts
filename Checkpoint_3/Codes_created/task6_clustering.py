#K. Anil Kumar
#GEID: 1011139542

import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
data_folder = os.getcwd()

#The product folder with csv files are generated from ner task
directory = input("Enter the path to the directory containing product csv files of all companies: ")
files = os.listdir(directory)
l=[]
p=[]
for i,f_name in enumerate(files):
    data=pd.read_csv(directory+"/"+f_name, sep=',', encoding='latin-1')
    for c in data.columns:
        p.append([])
        p[i].append(c)
    l.append(f_name.strip(".csv"))

p = [x for x in p if x != []]

for line in p:
    line.remove('quarter')

rows = zip(l, p)
with open('results\Products.csv', "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Company', 'Products'])
    for row in rows:
        writer.writerow(row)

    f.close()


data = pd.read_csv(r"results\Products.csv",error_bad_lines=False)
# data.head()

data[data['Products'].duplicated(keep=False)].sort_values('Products').head(8)


punc = ['.', ',', '"', '[', ']']
stop_words = text.ENGLISH_STOP_WORDS
desc = data["Products"].values
desc1=data["Company"].values
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

vectorizer1 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)
X1 = vectorizer1.fit_transform(desc1)
word_features2 = vectorizer1.get_feature_names()
#print(len(word_features2))

vectorizer2 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features = 1000)
X2 = vectorizer2.fit_transform(desc)
words = vectorizer2.get_feature_names()

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X2,data['Company'])
    wcss.append(kmeans.inertia_)

#Plot it and find k-value
'''plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()'''

# n_init(number of iterations for clustering) n_jobs(number of cpu cores to use)
kmeans = KMeans(n_clusters=5, n_init=20, n_jobs=1)
y = kmeans.fit_predict(X2, X1)
#print(y)
data['Cluster'] = y

li = []
common_words = kmeans.cluster_centers_.argsort()[:, -1:-20:-1]
for num, centroid in enumerate(common_words):
    li.append(str(num) + ' : ' + ' '.join(words[word] for word in centroid))

#print(data)
