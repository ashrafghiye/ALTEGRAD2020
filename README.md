# Altegrad challenge Fall 2020: H-index Prediction
 
**Team: Hear Me Roar**

- Ashraf GHIYE
- Hussein JAWAD

 
# Problem Description

The goal of this project is to study and apply machine learning techniques to a real-world regression problem. In this regression task, each sample corresponds to a researcher, i.e. an author of research papers, and the goal is to build a model that can predict accurately the h-index of each author. 
To build the model, we have access to two types of data: 

1. **Collaboration Graph** that models the collaboration intensity of two researchers: i.e. whether they have co-authored any papers.
2. **Abstracts** of the top-cited papers for each author.


# Motivation

Quantifying success in science plays a key role in guiding funding allocations, recruitment decisions, and rewards. Recently, a significant amount of progresses have been made towards quantifying success in science. 

We focus here on using artificielle intelligence to learn a predictive function based on two types of data that can assess the quality of a researcher's work (his abstracts) and the quantity of his work (collaboration intensity).

# Data

To get the data of this project you will have to run `get_data.sh`, all the necessary files will be downloaded in `data` folder.
These files include:
* `abstracts.txt`: for each paper, this file contains the ID of the paper along with the inverted index
of its abstract.
* `author_papers.txt`: list of authors and the IDs of their top-cited papers.
* `collaboration_network.edgelist`: a co-authorship network where nodes correspond to authors
that have published papers in computer science venues
* `train.csv`: 23,124 labeled authors. Each row of the file contains the ID of an author and
his/her h-index.
* `test.csv`: this file contains the IDs of 208,115 authors unlabelled.
* `paper_embeddings_64.csv`: Doc2Vec embeddings of each paper.
* `Text.json`: pre-processed texts for each abstract.

# Features

We have extracted some new features and created a new weighted graph of collaboration. To get all of these features into `features` folder you can run the script `feature_generators.sh` which extract the hand-crafted features explained in the report.

# Model

After you have run the previous two scripts, you can run `model.py` to train the model and see some benchmarks scores. After that an inference step is made by running `inference.py` which generates a `.csv` prediction for all the authors in the testing set.



