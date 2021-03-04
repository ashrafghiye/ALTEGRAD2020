import lightgbm as lgb
import pandas as pd
import numpy as np

"""
This script loads the pre-trained model and prepare the prediction.
"""

# read test data
df_test = pd.read_csv('data/test.csv')
n_test = df_test.shape[0]

# read features for each author of abstracts  
print("loading data ...")
df1=pd.read_csv("features/number_of_papers_features.csv")
df2=pd.read_csv("features/author_embedding.csv")
df3=pd.read_csv("features/deep_walk_features.csv")
df4=pd.read_csv("features/graph_features.csv")
df5=pd.read_csv("features/author_topics_stats.csv")
df6=pd.read_csv("features/author_abstract_stats.csv")
df7=pd.read_csv("features/line_features.csv")

#create the testing matrix
df_test = df_test.merge(df1, on="authorID")
df_test = df_test.merge(df2, on="authorID")
df_test = df_test.merge(df3, on="authorID", suffixes=("_author_embedding", "_deep_walk"))
df_test = df_test.merge(df4, on="authorID")
df_test = df_test.merge(df5, on="authorID")
df_test = df_test.merge(df6, on="authorID")
df_test = df_test.merge(df7, on="authorID")

X_test = df_test.drop(["authorID"],axis=1).values

print("done loading data...")
print("start inference...")


reg = lgb.Booster(model_file='mode.txt')
reg.fit(X_train, np.log10(y_train))
y_pred = 10**reg.predict(X_test)
df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
df_test.loc[:,["authorID","h_index_pred"]].to_csv('test_predictions.csv', index=False)

print("inference done...")
