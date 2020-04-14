#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:52:04 2020

@author: shivaborusu
"""

from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = load_wine()

x = pd.DataFrame(data=data.data, columns=data.feature_names )
y = pd.DataFrame(data=data.target, columns=['target'])

#for pearson correlation matrix for feature selection
fig, ax = plt.subplots(figsize=(8,8))  
sns.heatmap(pd.concat([x,y],axis=1).corr(), annot=True, vmin=-1,vmax=1, 
             fmt='.1g', linewidths=0.2, linecolor='black', ax=ax )


#pca application, feature scaling before PCA,this is feature extraction technique
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler(feature_range=[0,1])

x_scaled = scale.fit_transform(x)

pca = PCA()
pca_x = pca.fit(x_scaled)

plt.figure()
plt.plot(np.cumsum(pca_x.explained_variance_ratio_))
plt.plot([0,13],[0.4,1], 'r')
plt.xlim([0,14])
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

pca = PCA(n_components = 12)
pca_x = pca.fit(x_scaled)
pca_x = pca.transform(x_scaled)

x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.2, shuffle=True,
                                                    random_state = 42)

#selecting best features, this is feature selection technique
from sklearn.feature_selection import VarianceThreshold, SelectKBest



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True,
                                                    random_state = 42)

vt = VarianceThreshold().fit(x_train,y_train)

kbest = SelectKBest(k=8)
kb=kbest.fit(x_train,y_train)


#slectiong and fitting a model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


#model evaluation
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

accuracy_score(y_test, y_pred)

mat=confusion_matrix(y_test, y_pred, labels=[0,1,2])

fpr,tpr, thres =  roc_curve(y_test, model.predict_proba(x_test)[:,0], pos_label=0)
roc_auc = auc(fpr,tpr)

fpr1,tpr1, thres =  roc_curve(y_test, model.predict_proba(x_test)[:,1], pos_label=1)
roc_auc1 = auc(fpr,tpr)

fpr2,tpr2, thres =  roc_curve(y_test, model.predict_proba(x_test)[:,2], pos_label=2)
roc_auc2 = auc(fpr,tpr)


plt.subplots(figsize=(5,5))
plt.title("ROC AUC CURVE")
plt.plot(fpr,tpr, 'b', label='AUC0 = %0.2f' %roc_auc)
plt.plot(fpr1,tpr1, 'r', label='AUC1 = %0.2f' %roc_auc1)
plt.plot(fpr2,tpr2, 'g', label='AUC2 = %0.2f' %roc_auc2)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("FPR0")
plt.ylabel("TPR0")


#create a model pipeline now
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[('scaler', MinMaxScaler(feature_range=[0,1])),
                            ('pca', PCA(n_components = 12)),
                            ('model', DecisionTreeClassifier())
                            ])

pipeline.fit(x_train, y_train)

import pickle as pkl

pkl_file = open("class.pkl","wb")

pkl.dump(pipeline, pkl_file)

pkl_file.close()

#export test data as CSV for testing
x_test = pd.DataFrame(data=x_test, columns=data.feature_names, index=None)
x_test.to_csv('./x_test.csv', index=False)
