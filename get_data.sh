#!/bin/sh
cd data
gdown "https://drive.google.com/uc?id=1xFnj04M6GxTgjURDuuKC2VrGky0qxyB2"
gdown "https://drive.google.com/uc?id=1e1FQa4BD1FaaiFVlGfzW46QRP-PiN5pK"
gdown "https://drive.google.com/uc?id=1wc4rQt8UPwkLSB5IqeRIFGBZxAkBU2kd"
gdown "https://drive.google.com/uc?id=1G3aCGuKWLJr6kiF_eb2CzLhoq-DHLzlF"
gdown "https://drive.google.com/uc?id=1ahIDcG3CNUcAwOOy9iWx_NDcFErD5Lnn"
gdown "https://drive.google.com/uc?id=1-G5KjbCUcBcyPzLpsjEyBL3E0Tqwfi6E"
gdown "https://drive.google.com/uc?id=1oOjMApxN4OZZFPSHN56u9M9sqGantJi5"


python weighted_graph.py
python abstracts_word_counts.py

### you can download a 64 Doc2Vec embeddings with the above link, also here is the script to re-generate them.
#python paper_representations_64.py
