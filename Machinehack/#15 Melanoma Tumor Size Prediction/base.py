# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:51:28 2020

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
fea_int_cols = ['mass_npea', 'size_npear', 'damage_size',
       'exposed_area', 'std_dev_malign', 'err_malign', 'malign_penalty']

for i in fea_int_cols:
    data['mal_{}'.format(i)] = data['malign_ratio'] * data[i]

for i in fea_int_cols:
    data['dam_{}'.format(i)] = data['damage_ratio'] * data[i]
    
for i in list(fea_int_cols):
    data['size_npe_{}'.format(i)] = data['size_npear'] * data[i]
    
for i in list(fea_int_cols):
    data['size_npe_{}'.format(i)] = data['size_npear'] * data[i]

for i in list(fea_int_cols):
    data['dam_size_{}'.format(i)] = data['damage_size'] * data[i]


'''
from sklearn.preprocessing import MinMaxScaler1
scld = MinMaxScaler()'''

data.drop(columns = low_feat,inplace = True)

train_mod = data[data['tumor_size'] > -1]
test_mod = data[data['tumor_size'] == -1]
y = train_mod['tumor_size']
train_mod.drop(columns = ['tumor_size'],inplace = True)


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def rmse(model):
    return mean_squared_error(y_test,model.predict(X_test))


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_mod,y)
    
from sklearn.preprocessing import MinMaxScaler, StandardScaler
lasso = make_pipeline(StandardScaler(), Lasso(alpha =0.0005, random_state=1, max_iter=5000))
lasso.fit(X_train,y_train)
rmse(lasso)



ENet = make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=.5, random_state=3, max_iter=5000))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

from sklearn.kernel_ridge import KernelRidge
KRR = make_pipeline(StandardScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
rf.fit(X_train,y_train)
rmse(rf)

low_feat = list(coef.tail(15).index)




















rf.fit(df_dummies_train, y.ravel())
# Output feature importance coefficients, map them to their feature name, and sort values
coef = pd.Series(rf.feature_importances_, index = X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(20, 10))
coef.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()

sub = sample.copy()
sub['tumor_size'] = rf.predict(test_mod.drop(columns = 'tumor_size'))
sub.to_csv('rfr2.csv' , index = False)





