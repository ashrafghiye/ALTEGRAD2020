import networkx as nx
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ast
import re
import community
import itertools



#read the collaboration graph
G = nx.read_edgelist("data/collaboration_network.edgelist")
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

# read the file to create a dictionary with author key and paper list of each author as value
f = open("data/author_papers.txt","r")
papers_set = set()
d = {}
for l in f:
    auth_paps = [paper_id.strip() for paper_id in l.split(":")[1].replace("[","").replace("]","").replace("\n","").replace("\'","").replace("\"","").split(",")]
    d[l.split(":")[0]] = auth_paps    
f.close()

#set initial weights to 1.0, suppose that at least they worked 1.0
nx.set_edge_attributes(G, values = -1.0, name = 'weight')

#well, sitll an experiment.. weighted graph with weights = 1/number of collaborations
nodes = set()
papers = {}

for author_id in d:
    for paper in d[author_id]:
        if paper in papers:
            papers[paper].append(author_id)
        else: 
            papers[paper] = [author_id]        


for paper in papers:
    L = papers[paper]
    if len(L)>1:
        edges = itertools.combinations(L,2)
        for v1, v2 in list(edges):
            if G.has_edge(v1, v2):
                if G[v1][v2]['weight'] < 0:
                    G[v1][v2]['weight'] = 1.0/(len(L)-1)
                else:
                    G[v1][v2]['weight'] += 1.0/(len(L)-1)
            else:
                G.add_edge(v1, v2, weight=1.0/(len(L)-1))

for e in G.edges():
    if G[e[0]][e[1]]['weight'] < 0:
        G[e[0]][e[1]]['weight'] = 1.0 #supposing the paper has 2 authors.. (strongest relation)

G.remove_edges_from(nx.selfloop_edges(G))
nx.write_weighted_edgelist(G, "data/denser_graph.edgelist")