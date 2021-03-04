import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb

# read training data
df_train = pd.read_csv('data/train.csv', dtype={'h_index': np.float32})
n_train = df_train.shape[0]

# read features for each author of abstracts  
print("loading data ...")
df1=pd.read_csv("features/number_of_papers_features.csv")
df2=pd.read_csv("features/author_embedding.csv")
df3=pd.read_csv("features/deep_walk_features.csv")
df4=pd.read_csv("features/graph_features.csv")
df5=pd.read_csv("features/author_topics_stats.csv")
df6=pd.read_csv("features/author_abstract_stats.csv")
df7=pd.read_csv("features/line_features.csv")

# create the training matrix. each author is represented by the features that we already calculated 
df_train = df_train.merge(df1, on="authorID")
df_train = df_train.merge(df2, on="authorID")
df_train = df_train.merge(df3, on="authorID", suffixes=("_author_embedding", "_deep_walk"))
df_train = df_train.merge(df4, on="authorID")
df_train = df_train.merge(df5, on="authorID")
df_train = df_train.merge(df6, on="authorID")
df_train = df_train.merge(df7, on="authorID")

X_train = df_train.drop(["authorID","h_index"],axis=1).values
y_train = df_train["h_index"].values

print("done loading data...")
print("start modeling...")

X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.25, random_state=42) 
y_tr_log = np.log10(y_tr)
# train a regression model and make predictions
# using lasso
print("prediction using Lasso:")
reg = make_pipeline(StandardScaler(), LassoCV())
reg.fit(X_tr, y_tr)
y_pred_train = reg.predict(X_tr)
y_pred_test = reg.predict(X_va)
print("Lasso mean absolute error on train :",mean_absolute_error(y_tr, y_pred_train))
print("Lasso mean absolute error on test :",mean_absolute_error(y_va, y_pred_test))
print("prediction using Lasso with log y_train:")
reg = make_pipeline(StandardScaler(), LassoCV())
reg.fit(X_tr, y_tr_log)
y_pred_train = 10**reg.predict(X_tr)
y_pred_test = 10**reg.predict(X_va)
print("Lasso mean absolute error on train with log y_train :",mean_absolute_error(y_tr, y_pred_train))
print("Lasso mean absolute error on test with log y_train :",mean_absolute_error(y_va, y_pred_test))
##
## using KNN
print("prediction using KNN:")
reg = KNeighborsClassifier(n_neighbors=9)
reg.fit(X_tr, y_tr)
y_pred_train = reg.predict(X_tr)
y_pred_test = reg.predict(X_va)
print("KNN mean absolute error on train :",mean_absolute_error(y_tr, y_pred_train))
print("KNN mean absolute error on test :",mean_absolute_error(y_va, y_pred_test))
print("prediction using KNN with log y_train:")
reg = make_pipeline(StandardScaler(), LassoCV())
reg.fit(X_tr, y_tr_log)
y_pred_train = 10**reg.predict(X_tr)
y_pred_test = 10**reg.predict(X_va)
print("KNN mean absolute error on train with log y_train :",mean_absolute_error(y_tr, y_pred_train))
print("KNN mean absolute error on test with log y_train :",mean_absolute_error(y_va, y_pred_test))
##
## using Random Forest
print("prediction using Random Forest Regressor:")
reg = RandomForestRegressor()
reg.fit(X_tr, y_tr)
y_pred_train = reg.predict(X_tr)
y_pred_test = reg.predict(X_va)
print("Random Forest Regressor mean absolute error on train :",mean_absolute_error(y_tr, y_pred_train))
print("Random Forest Regressor mean absolute error on test :",mean_absolute_error(y_va, y_pred_test))
print("prediction using Random Forest Regressor with log y_train:")
reg = RandomForestRegressor()
reg.fit(X_tr, y_tr_log)
y_pred_train = 10**reg.predict(X_tr)
y_pred_test = 10**reg.predict(X_va)
print("Random Forest Regressor mean absolute error on train with log y_train:",mean_absolute_error(y_tr, y_pred_train))
print("Random Forest Regressor mean absolute error on test with log y_train:",mean_absolute_error(y_va, y_pred_test))
##
## using xgboost
print("prediction using xgboost:")
reg = xgb.XGBRegressor(n_estimators = 50)
reg.fit(X_tr, y_tr)
y_pred_train = reg.predict(X_tr)
y_pred_test = reg.predict(X_va)
print("XGBoost mean absolute error on train :",mean_absolute_error(y_tr, y_pred_train))
print("XGBoost mean absolute error on test :",mean_absolute_error(y_va, y_pred_test))
print("prediction using xgboost with log y_train:")
reg = xgb.XGBRegressor(n_estimators = 50)
reg.fit(X_tr, y_tr_log)
y_pred_train = 10**reg.predict(X_tr)
y_pred_test = 10**reg.predict(X_va)
print("XGBoost mean absolute error on train with log y_train :",mean_absolute_error(y_tr, y_pred_train))
print("XGBoost mean absolute error on test with log y_train :",mean_absolute_error(y_va, y_pred_test))
##
## using Lightgbm
print("prediction using Lightgbm:")
from sklearn.model_selection import StratifiedKFold
import math

clf = lgb.LGBMRegressor(is_unbalance=True,
                        colsample_bytree=0.4, 
                        importance_type='gain', alpha=0.05,
                        objective='rmse', learning_rate=0.025,
                        n_estimators=6000,
                        )


skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(X_train, y_train)

scores = []
scores_train = []

for train_index, test_index in skf.split(X_train, y_train):
    print("TRAIN:", train_index, "TEST:", test_index);
    X_tr, X_val = X_train[train_index], X_train[test_index]
    y_tr, y_val = y_train[train_index], y_train[test_index]
    clf.fit(X_tr, y_tr)
    scores.append(np.mean(np.abs(y_val - clf.predict(X_val))))
    scores_train.append(np.mean(np.abs(y_tr - clf.predict(X_tr))))
    
  

print("Lightgbm mean absolute error on train :",np.mean(scores_train))
print("Lightgbm mean absolute error on test :",np.mean(scores))

print("prediction using Lightgbm with log y_train:")

clf = lgb.LGBMRegressor(is_unbalance=True,
                        colsample_bytree=0.4, 
                        importance_type='gain', alpha=0.05,
                        objective='mse', learning_rate=0.025,
                        n_estimators=6000,
                        )


skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(X_train, y_train)

alpha = 10
scores = []

for train_index, test_index in skf.split(X_train, y_train):
    print("TRAIN:", train_index, "TEST:", test_index);
    X_tr, X_val = X_train[train_index], X_train[test_index]
    y_tr, y_val = y_train[train_index], y_train[test_index]
    clf.fit(X_tr, np.array([math.log(y, alpha) for y in y_tr]))
    scores.append(np.mean(np.abs(y_val - alpha**clf.predict(X_val))))
    scores_train.append(np.mean(np.abs(y_tr - clf.predict(X_tr))))
  
print("prediction using Lightgbm with log y_train:")
  
print("Lightgbm mean absolute error on train with log y_train :",np.mean(scores_train))
print("Lightgbm mean absolute error on test with log y_train :",np.mean(scores))

clf.fit(X_train, y_train)
print("Training done.")

#load from model:
clf.booster_.save_model('mode.txt')
#bst = lgb.Booster(model_file='mode.txt') to load later
print("Model saved.")
