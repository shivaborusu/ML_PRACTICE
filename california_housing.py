#!/usr/bin/env python
# coding: utf-8


from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle as pkl
from flask import Flask
from sklearn.preprocessing import FunctionTransformer
from select1 import select_features



dataset = fetch_california_housing()


df = pd.DataFrame(dataset.data, columns=dataset.feature_names)


df_target  = pd.DataFrame(data=dataset.target, columns=["target"])


total_df = pd.concat([df,df_target], axis=1, sort=False)


sns.heatmap(total_df.corr(), annot = True, fmt = '.1g', vmin=-1, vmax=1, center=0, 
            cmap='summer', linewidths= 2, linecolor='black')


new_df = total_df[['MedInc', 'HouseAge', 'AveRooms','Latitude', 'Longitude']]


new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_df, df_target, random_state=42,
                                                                    test_size = 0.2)

pipe = Pipeline(steps=[('scale', StandardScaler()), ('model_1', DecisionTreeRegressor(criterion = 'mae'))])

pipe2 = Pipeline(steps=[('scale', StandardScaler()), ('model_2', LinearRegression())])


pipe3 = Pipeline(steps=[('scale', StandardScaler()),('poly', PolynomialFeatures(degree=3)),
                        ('model_3', LinearRegression())])

pipe4 = Pipeline(steps=[('scale', StandardScaler()),('poly', PolynomialFeatures(degree=3)),
                        ('model_4', DecisionTreeRegressor(criterion='mae'))])

pipe5 = Pipeline(steps=[('scale', StandardScaler()),('poly', PolynomialFeatures(degree=3)),
                        ('model_5', RandomForestRegressor(n_estimators=3, criterion='mae'))])

pipe5.fit(new_x_train,new_y_train.values.ravel())

model_7_op = pipe5.predict(new_x_test)

r2_score(new_y_test, model_7_op)

mean_absolute_error(new_y_test, model_7_op)

#code to productionize
#try to fit this on the rawdata after splitting
x_train, x_test, y_train, y_test = train_test_split(df, df_target, random_state=42,
                                                                    test_size = 0.2)

#creating a pipeline with features selection on the production data, or test data
def select_features(df):
    return df[['MedInc', 'HouseAge', 'AveRooms','Latitude', 'Longitude']]

# pipeline to get all tfidf and word count for first column
pipeline_final = Pipeline(steps=[
    ('column_selection', FunctionTransformer(select1.select_features, validate=False)),
    ('scale', StandardScaler()),('poly', PolynomialFeatures(degree=2)),
                        ('model_5', RandomForestRegressor(n_estimators=3, criterion='mae'))])
    
#fitting preprocessing and model to the training dataset    
pipeline_final.fit(x_train,y_train)


output = pipeline_final.predict(x_test)


#validation
error = mean_absolute_error(y_test, output)

r2 = r2_score(y_test, output)

#pickling
pkl_file = open("pkl_file.pkl", "wb")
pkl.dump(pipeline_final,pkl_file)
pkl_file.close()


