import networkx as nx
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ast
import re
import community
from easygraph.functions.graph_embedding.line import LINE

G = nx.read_edgelist('../data/denser_graph.edgelist', nodetype=str, data=(("weight", float), ))
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

n_dim = 16


    
model = LINE(G, negative_ratio=10)
model.train(batch_size=5000, epochs=3, verbose=1)
embeddings = model.get_embeddings() # Returns the graph embedding results

features_line = np.zeros((n_nodes, n_dim))
for i, node in enumerate(G.nodes()):
    features_line[i,:] = embeddings[str(node)]

df = pd.DataFrame(features_line)
df["authorID"] = list(G.nodes())
df.to_csv("../features/line_features.csv",index=False)


