import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec
import pandas as pd
"""
This script Generate a deepwalk features for each author based on the graph in the dataset
"""
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):
    
    ##################
    walk = [node]
    for i in range(walk_length):
        node = np.random.choice(list(G.neighbors(node)))
        walk.append(node)
    ##################

    walk = [str(node) for node in walk]
    return walk


# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    ##################
    nodes = list(G.nodes)
    for i in range(num_walks):
        shuffled_nodes = np.random.permutation(nodes)
        walks += [random_walk(G, node, walk_length) for node in shuffled_nodes]
    ##################

    return walks


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model


G = nx.read_edgelist('../data/collaboration_network.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)
n = G.number_of_nodes()

############## Task 6
n_dim = 64
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

df = pd.DataFrame(embeddings)
df["authorID"] = list(G.nodes())

df.to_csv("../features/deep_walk_features.csv",index=False)
