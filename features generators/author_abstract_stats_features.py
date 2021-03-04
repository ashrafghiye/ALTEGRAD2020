"""
This code will extract some statisical features for each author, based on the length of its papers

It will take as inputs files : 
- author_papers.txt
- abstract_word_count.csv

It will output one file:
- author_abstract_stats
"""

import ast
import re

import numpy as np
import pandas as pd
import scipy.stats as stat


# read the file to create a dictionary with authorID as key and papersID list as value
f = open("../data/author_papers.txt", "r")
papers_set = set()
author_papers_d = {}
for line in f:
    auth_paps = [paper_id.strip() for paper_id in line.split(":")[1].replace("[", "").replace("]", "").replace("\n", "").replace("\'", "").replace("\"", "").split(",")]
    author_papers_d[line.split(":")[0]] = auth_paps
f.close()

# now, read length of each paper abstract
abstract_length = pd.read_csv("../data/abstract_word_count.csv", usecols=['paper_id', 'word_count'])
abstract_length = {paper_id:word_count for paper_id, word_count in zip(abstract_length['paper_id'], abstract_length['word_count'])}

author_abstract_stats=pd.DataFrame()

authorID = list(author_papers_d.keys())
mean_length = []
min_length = []
max_length = []
mode_length = []

for author in authorID:
    
    papers = author_papers_d[author]
    v = []
    for paper in author_papers_d[author]:
        try:
            v.append(abstract_length[int(paper)])
        except:
            continue
        
    if(len(v)==0):
        v = [0]
        
    mean_length.append(np.mean(v))
    min_length.append(np.min(v))
    max_length.append(np.max(v))
    mode_length.append(stat.mode(v)[0][0])

author_abstract_stats["authorID"] = authorID
author_abstract_stats["mean_length"] = mean_length       
author_abstract_stats["min_length"] = min_length       
author_abstract_stats["max_length"] = max_length
author_abstract_stats["mode_length"] = mode_length       

author_abstract_stats.to_csv("../features/author_abstract_stats.csv",index=False)
