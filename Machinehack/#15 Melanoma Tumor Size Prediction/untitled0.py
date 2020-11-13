# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 23:47:05 2020

@author: dineshy86
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample= pd.read_csv('sample.csv')

corr = train.corr()

train.columns
test['tumor_size'] = -1

data = pd.concat([train,test])
'''
from sklearn.preprocessing import MinMaxScaler1
scld = MinMaxScaler()'''

train_mod = data[data['tumor_size'] > -1]
test_mod = data[data['tumor_size'] == -1]
y = train_mod['tumor_size']
train_mod.drop(columns = ['tumor_size'],inplace = True)





import lightgbm as lgb


# custom function to run light gbm model
params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : 30,
    "min_child_samples" : 100,
    "learning_rate" : 0.1,
    "bagging_fraction" : 0.7,
    "feature_fraction" : 0.5,
    "bagging_frequency" : 5,
    "bagging_seed" : 2018,
    "verbosity" : -1
}

lgtrain = lgb.Dataset(X_train, label=y_train)
lgval = lgb.Dataset(X_test, label=y_test)
model = lgb.train(params, lgtrain, 5000, valid_sets=[lgval], verbose_eval=100)

lgb.LGBMRegressor()
def rmse(model):
    return mean_squared_error(y_test,model.predict(X_test))

import xgboost
xgbr = xgboost.XGBRegressor(max_depth = 7,min_child_weight=1,n_jobs = -1,n_estimators= 1000)
xgbr.fit(X_train,y_train)
rmse(xgbr)



  
estimators = [('lr', RidgeCV()),
    ('xgb', xgbr,)
    ('light',lgtrain),
    

]





