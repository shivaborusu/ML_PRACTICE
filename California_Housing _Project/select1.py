#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:24:54 2020

@author: shivaborusu
"""

#creating a pipeline with features selection on the production data, or test data
def select_features(df):
    return df[['MedInc', 'HouseAge', 'AveRooms','Latitude', 'Longitude']]