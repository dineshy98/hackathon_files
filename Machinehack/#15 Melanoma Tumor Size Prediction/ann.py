# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 23:54:33 2020

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




from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping


model = Sequential()
model.add(Dense(35,input_shape = (29,),activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation = 'linear'))

model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.fit(X_train,y_train,epochs = 20)

