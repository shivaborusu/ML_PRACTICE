# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.decomposition import PCA, KernelPCA

data = load_breast_cancer()

x = pd.DataFrame(data=data.data, columns = data.feature_names)
y = pd.DataFrame(data=data.target, columns = ['Target'])

#removing low varinace featurs, varinces are calculated in individual columns
vt = VarianceThreshold(threshold=1)
variances = vt.fit(X=x,y=None)
x_var = vt.transform(x)

#study the correlations the data
plt.subplots(figsize=(6,6))
sns.heatmap(pd.DataFrame(x_var).corr(), annot = True, vmin = -1, vmax = 1, 
            cbar=True)


#scaling the data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale_x = scale.fit_transform(x_var)

x_np = scale_x[:, [0,1,2,3,4,5]]

#study the correlations the data
plt.subplots(figsize=(6,6))
sns.heatmap(pd.DataFrame(x_np).corr(), annot = True, vmin = -1, vmax = 1, 
            cbar=True)

#fit a model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=5, criterion = 'gini', random_state = 42)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_np, y, test_size=0.3, 
                                                    random_state=42,shuffle = True)

model = rf.fit(x_train, y_train.values.ravel())

y_pred = rf.predict(x_test)



#evaluate
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.metrics import classification_report

mat = confusion_matrix(y_test, y_pred)


#eventho accuracy is 94%, try with grid search and cross validation
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

rfc = RandomForestClassifier()

from sklearn.tree import DecisionTreeClassifier
dt =  DecisionTreeClassifier()

param_grid = {'n_estimators': [2,3,4,5,6], 'criterion': ['gini', 'entropy']}

model_grid = GridSearchCV(rfc, param_grid = param_grid)

model_grid.fit(y_test, y_pred)

#as per grid, gini, n_estimator=2 are best

rfc1 = RandomForestClassifier(n_estimators=2, criterion='gini', random_state=42)
kf = KFold(shuffle=True, n_splits=5)
model_cv = cross_val_score(estimator = rfc, X=x_train, 
                           y=y_train.values.ravel(), cv=kf)
