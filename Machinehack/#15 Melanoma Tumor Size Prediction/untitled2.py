# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 01:18:40 2020

@author: dineshy86
"""





from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR,NuSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor



  
estimators = [('lr', RidgeCV()),
    ('svr0', SVR(kernel = 'linear')),
    ('svr1', SVR(kernel = 'poly')),
    ('svr2', SVR(kernel = 'sigmoid')),
    ('svr3', SVR(kernel = 'rbf')),
    ('svr4', NuSVR(kernel = 'linear')),
    ('svr5', NuSVR(kernel = 'poly')),
    ('svr6', NuSVR(kernel = 'sigmoid')),
    ('svr7', NuSVR(kernel = 'rbf')),
    ('knn1', KNeighborsRegressor(n_neighbors = 5)),
    ('knn2', KNeighborsRegressor(n_neighbors = 10)),
    ('rnn1', RadiusNeighborsRegressor(radius = 1)),
    ('rnn2', RadiusNeighborsRegressor(radius = 2)),
    ('dtr1', DecisionTreeRegressor(splitter = 'best')),
    ('dtr2', DecisionTreeRegressor(splitter = 'random'))

]

reg = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(n_estimators=1000),
    passthrough = True,
    verbose = 50,
    n_jobs = -1
)

reg.

reg.fit(X_train, y_train)





