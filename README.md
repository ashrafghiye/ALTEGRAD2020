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

# Getting Started

## Download Data


To get the data of this project you will have to run `get_data.sh`.

> `chmod +x get_data.sh`
> 
> `./get_data.sh`

All the necessary files will be downloaded in `data` folder, these files include:

* `abstracts.txt`: for each paper, this file contains the ID of the paper along with the inverted index
of its abstract. Roughly speaking, an inverted index is a dictionary where the keys correspond
to words and the values are lists of the positions of the corresponding words in the abstract. For
example, the abstract “that is all that I know” would be represented as follows:

> paper_ID----{"IndexLength":5,"InvertedIndex": {"that":[0,3],"is":[1],"all":[2],"I":[4],"know":[5]}}

* `author_papers.txt`: list of authors and the IDs of their top-cited papers.
* `collaboration_network.edgelist`: a co-authorship network where nodes correspond to authors
that have published papers in computer science venues
* `train.csv`: 23,124 labeled authors. Each row of the file contains the ID of an author and
his/her h-index.
* `test.csv`: this file contains the IDs of 208,115 authors unlabelled.
* `paper_embeddings_64.csv`: Doc2Vec embeddings of each paper.
* `Text.json`: plain texts for each abstract. The format is paperID: abstract in plain text, it is useful to performe NLP downstream tasks on this file like text embeddings.

## Install the requirements

>  `pip install -U -r requirements.txt`

## Generate Features

We have extracted some new features and created a new weighted graph of collaboration. To get all of these features into `features` folder you can run the script `feature_generators.sh` which extract the hand-crafted features explained in the report.

> `chmod +x feature_generators.sh`
> 
> `./feature_generators.sh`

## Training 

After you have run the previous two scripts, you can run `model.py` to train the model and see some benchmarks scores. For more details about the model and other benchmarks refer to `final_report.pdf`.

## Run a test

After training the model we can run an inference step is made by running `inference.py` which generates a `.csv` prediction for all the authors in the testing set.



