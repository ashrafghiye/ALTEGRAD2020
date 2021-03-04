import json 
import pandas as pd

"""
This file caculate the len of each abstracts based on word count
"""
# Opening JSON file 
with open('data/Text.json') as json_file: 
    dic = json.load(json_file)

docs = list(dic.values())
keys = list(dic.keys())

Text_dataframe = pd.DataFrame(
    {'paper_id': keys,
     'Text': docs,
    })

word_counts = pd.DataFrame()
word_counts["paper_id"] = keys
word_counts["word_count"] = [len(words.split(" ")) for words in docs]
word_counts.to_csv("data/abstract_word_count.csv", index=False)
