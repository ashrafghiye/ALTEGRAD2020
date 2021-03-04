"""
This code generate features based on the topic of papers that the author has wrote.

It take two files:
- paper_embeddings_64.txt
- author_papers.txt

Then we will apply clustering on the embeddings of the papers in such way that we group papers that have similar topic (Before doing so, we first need to lower the dimensionality of the embeddings as many clustering algorithms handle high dimensionality poorly.)

After clustering papers, we give label to each cluster, therefore each label will  represent one topic.

Next, for each author, we will generate some features based on the topic of its papers.

This code will output one file:
- author_topics_stats.csv

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import ast
import re
from sklearn.cluster import Birch
import scipy.stats as stat


###########################Predict the topic of each paper#########################################################

#load papers embeddings in order to do clustring
f = open("../data/paper_embeddings_64.txt","r")
papers = {}
s = ""
pattern = re.compile(r'(\s){2,}')
for l in f:
    if(":" in l and s!=""):
        papers[s.split(":")[0]] = np.array(ast.literal_eval(re.sub(pattern, ',', s.split(":")[1]).replace(" ",",")))
        s = l.replace("\n","")
    else:
        s = s+" "+l.replace("\n","")
    
f.close()

papers_id  = list(papers.keys())
embeddings = list(papers.values())

#umap dimensionality reduction
umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, 
                            metric='euclidean').fit_transform(embeddings)

#clustring using Birch algorithm
cluster = Birch(n_clusters=32).fit(umap_embeddings)

Topic_papers = pd.DataFrame()
Topic_papers["paper_id"]=papers_id
Topic_papers["Topic"]=cluster.labels_
Topic_papers.to_csv("data/topic_papers.csv", index=False)
###################################################################################################################



###########################Extract features for each author from the topics of its paper###########################

# read the file to create a dictionary with authorID as key and papersID list as value
f = open("../data/author_papers.txt", "r")
papers_set = set()
author_papers_d = {}
for line in f:
    auth_paps = [paper_id.strip() for paper_id in line.split(":")[1].replace("[", "").replace("]", "").replace("\n", "").replace("\'", "").replace("\"", "").split(",")]
    author_papers_d[line.split(":")[0]] = auth_paps
f.close()

# now, read the Topic of each paper
paper_topics = {paper_id:topic for paper_id, topic in zip(Topic_papers['paper_id'], Topic_papers['Topic'])}

#create a file contains the features for each authors based on topics of its papers
#features:
#author disciplines (diversity or unique topics)
#author diversity ratio = unique topics / number of papers
#author major topic (mode of his papers topic)
#author contirbution in his major topic
#author contirbution ratio = author contirbution/ number of papers

df = open("../features/author_topics_stats.csv", "w")

s = "authorID" + "," + ",".join(["diversity", "diversity_ratio", "major_topic", "contribution", "contribution_ratio"])
df.write(s+"\n")

for author in author_papers_d:
    v = []
    c = 0
    for paper in author_papers_d[author]:
        try:
            v.append(paper_topics[paper])
            c += 1
        except:
            continue
            
    if c == 0:
        v = np.array([0, 0, 0, 0, 0])
    else:
        uni_len = len(np.unique(v))
        max_freq = np.max(np.unique(v, return_counts=True)[1])
        v = np.array([uni_len, uni_len/c, stat.mode(v)[0][0], max_freq, max_freq/c])
        
    df.write(author + "," + ",".join(map(lambda x: "{:.8f}".format(round(x, 8)), v)) + "\n")
df.close()
#######################################################################################################################
