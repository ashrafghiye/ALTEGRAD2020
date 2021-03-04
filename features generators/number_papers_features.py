import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ast
import re
"""
This code create features for each author based on its number of papers and the number of papers for its neighbours

It take as input two files:
- collaboration_network.edgelist
- author_papers.txt 

It ouputs one file:
- number_of_papers_features.csv
"""
#read the graph file to find the neighbours for each author
G = nx.read_edgelist('../data/collaboration_network.edgelist', delimiter=' ', nodetype=str)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

# read the file to create a dictionary with author key and paper list as value
f = open("../data/author_papers.txt","r")
papers_set = set()
d = {}
for l in f:
    auth_paps = [paper_id.strip() for paper_id in l.split(":")[1].replace("[","").replace("]","").replace("\n","").replace("\'","").replace("\"","").split(",")]
    d[l.split(":")[0]] = auth_paps
f.close()

# Create features 
authorID = list(d.keys())
number_of_papers = [len(d[author]) for author in authorID]

mean_G_neigh_papers = []
max_G_neigh_papers = []
min_G_neigh_papers = []

for node in authorID:
    
    neigh_papers = [G.degree(n) for n in G.neighbors(node)]

    mean_G_neigh_papers.append(np.mean(neigh_papers))
    max_G_neigh_papers.append(np.min(neigh_papers))
    min_G_neigh_papers.append(np.max(neigh_papers))

data = pd.DataFrame()
data["authorID"] = authorID
data["number_of_papers"] = number_of_papers
data["mean_G_neigh_papers"] = mean_G_neigh_papers
data["max_G_neigh_papers"] = max_G_neigh_papers
data["min_G_neigh_papers"] = min_G_neigh_papers

data.to_csv("../features/number_of_papers_features.csv", index=False)
