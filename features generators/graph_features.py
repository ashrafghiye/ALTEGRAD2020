import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy import sparse
from random import randint
from sklearn.cluster import KMeans

"""
This script Generate features for each author based on the 2 graphes in the dataset:
- the weighted graph 
- the unweighted graph

for each node, it calculate the following features:
1- degree in the unweighted graph
2- degree in the weighted graph
3- core number in the unweighted graph
4- core number in the weighted graph
5- the centrality in the unweighted graph
6- the centrality in the weighted graph
7- log10 of the pagerank in the unweighted graph
8- log10 of the pagerank in the weighted graph
9- spectral clustering coefficient in the unweighted graph
10- clustering coefficient in the unweighted graph
11- betweeness coefficient in the unweighted graph
12- mean of the neighbours degree in unweighted graph 
13- max of the neighbours degree in unweighted graph 
14- min of the neighbours degree in unweighted graph 
15- mean of the neighbours degree in weighted graph 
16- max of the neighbours degree in weighted graph 
17- min of the neighbours degree in weighted graph
18- mean of the neighbours page rank in unweighted graph 
19- max of the neighbours page rank in unweighted graph 
20- min of the neighbours page rank in unweighted graph
21- mean of the neighbours page rank in weighted graph 
22- max of the neighbours page rank in weighted graph 
23- min of the neighbours page rank in weighted graph

It take as input two files:
- collaboration_network.edgelist
- new_weighted.edgelist 

It ouputs one file:
- graph_features.csv

"""

def spectral_clustering(G, k):
    """
    :param G: networkx graph
    :param k: number of clusters
    :return: clustering labels, dictionary of nodeID: clustering_id
    """
    A = nx.adjacency_matrix(G)
    d = np.array([G.degree(node) for node in G.nodes()])
    D_inv = sparse.diags(1 / d)
    n = len(d)  # number of nodes
    L_rw = sparse.eye(n) - D_inv @ A

    eig_values, eig_vectors = eigs(L_rw, k, which='SR')
    eig_vectors = eig_vectors.real

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(eig_vectors)

    clustering_labels = {node: kmeans.labels_[i] for i, node in enumerate(G.nodes())}

    return clustering_labels


# load the graph    
G = nx.read_edgelist('data/collaboration_network.edgelist', delimiter=' ', nodetype=int)
WG = nx.read_edgelist('data/denser_graph.edgelist', nodetype=int, data=(("weight", float), ))

n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

# computes structural features for each node
core_number = nx.core_number(G)
core_number_w = nx.core_number(WG)
spectral_c = spectral_clustering(G, 10)
clustering_coef = nx.clustering(WG)
betweeness_coef = nx.betweenness_centrality(G, k=256)
centrality = nx.eigenvector_centrality(G)
centrality_w = nx.eigenvector_centrality(WG)
print("Centrality measures generated")

# papers count dictionary, for each author how many papers does he wrote.
df_papers = pd.read_csv('features/number_of_papers_features.csv', dtype={'authorID': str})
papers_dict = {key: val for key, val in zip(df_papers['authorID'].values, df_papers['number_of_papers'].values)}


degrees = np.array([G.degree(node) for node in G.nodes()])
degrees = degrees / np.sum(degrees)

degrees_w = np.array([WG.degree(node) for node in WG.nodes()])
degrees_w = degrees_w / np.sum(degrees_w)

papers_w = np.array([papers_dict[str(node)] for node in WG.nodes()])
papers_w = papers_w / np.sum(papers_w)


p_v = {node:degree for node,degree in zip(G.nodes(), degrees)}
p_v2 = {node: paperc for node,paperc in zip(WG.nodes(), papers_w)}
p_wv = {node:degree for node,degree in zip(WG.nodes(), degrees_w)}

pr = nx.pagerank(G, alpha=0.85)
wpr = nx.pagerank(WG, alpha=0.7, personalization=p_v2)
print("PageRank generated")

Graph_features = pd.DataFrame()

nodes = list(G.nodes())
Graph_features["authorID"] = nodes
Graph_features["G_degree"] = [G.degree(node) for node in nodes]
Graph_features["WG_degree"] = [WG.degree(node) for node in nodes]
Graph_features["G_core_number"] = [core_number[node] for node in nodes]
Graph_features["WG_core_number"] = [core_number_w[node] for node in nodes]
Graph_features["G_centrality"] = [centrality[node] for node in nodes]
Graph_features["WG_centrality"] = [centrality_w[node] for node in nodes]
Graph_features["G_page_rank"] = [(pr[node]) for node in nodes]
Graph_features["WG_page_rank"] = [(wpr[node]) for node in nodes]
Graph_features["G_spectral_c"] = [spectral_c[node] for node in nodes]
Graph_features["G_clustering_coef"] = [clustering_coef[node] for node in nodes]
Graph_features["G_betweeness_coef"] = [betweeness_coef[node] for node in nodes]

mean_G_neigh_degree = []
max_G_neigh_degree = []
min_G_neigh_degree = []

mean_WG_neigh_degree = []
max_WG_neigh_degree = []
min_WG_neigh_degree = []

mean_G_pg = []
max_G_pg = []
min_G_pg = []

mean_WG_pg = []
max_WG_pg = []
min_WG_pg = []

for node in nodes:

    neigh_degree = [G.degree(n) for n in G.neighbors(node)]
    neigh_W_degree = [WG.degree(n) for n in WG.neighbors(node)]
    neigh_pg = [pr[n] for n in G.neighbors(node)]
    neigh_W_pg = [wpr[n] for n in WG.neighbors(node)]

    mean_G_neigh_degree.append(np.mean(neigh_degree))
    max_G_neigh_degree.append(np.min(neigh_degree))
    min_G_neigh_degree.append(np.max(neigh_degree))

    mean_WG_neigh_degree.append(np.mean(neigh_W_degree))
    max_WG_neigh_degree.append(np.min(neigh_W_degree))
    min_WG_neigh_degree.append(np.max(neigh_W_degree))
  
    mean_G_pg.append(np.mean(neigh_pg))
    max_G_pg.append(np.min(neigh_pg))
    min_G_pg.append(np.max(neigh_pg))

    mean_WG_pg.append(np.mean(neigh_W_pg))
    max_WG_pg.append(np.min(neigh_W_pg))
    min_WG_pg.append(np.max(neigh_W_pg))
    

Graph_features["mean_G_neigh_degree"] = mean_G_neigh_degree
Graph_features["max_G_neigh_degree"] = max_G_neigh_degree
Graph_features["min_G_neigh_degree"] = min_G_neigh_degree

Graph_features["mean_WG_neigh_degree"] = mean_WG_neigh_degree
Graph_features["max_WG_neigh_degree"] = max_WG_neigh_degree
Graph_features["min_WG_neigh_degree"] = min_WG_neigh_degree

Graph_features["mean_G_pg"] = mean_G_pg
Graph_features["max_G_pg"] = max_G_pg
Graph_features["min_G_pg"] = min_G_pg

Graph_features["mean_WG_pg"] = mean_WG_pg
Graph_features["max_WG_pg"] = max_WG_pg
Graph_features["min_WG_pg"] = min_WG_pg

print("Neighbours features generated")

Graph_features.to_csv("features/graph_features.csv",index=False)

print("Done")