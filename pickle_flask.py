#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:41:59 2020

@author: shivaborusu
"""

#california data set unpickling and flask
import pickle as pkl
from flask import Flask, request
import pandas as pd

infile = open("./pkl_file.pkl","rb")

model = pkl.load(infile)

infile.close()

app = Flask(__name__)

@app.route('/predict')
def predict_price():
    ''' MedInc = request.args.get("MedInc")
    HouseAge = request.args.get("HouseAge")
    AveRooms = request.args.get("AveRooms")
    AveBedrms = request.args.get("AveBedrms")
    Population = request.args.get("Population")
    AveOccup = request.args.get("AveOccup")
    Latitude = request.args.get("Latitude")
    Longitude = request.args.get("Longitude")
    
    lst = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]'''
    lst = [[1, 2, 3, 4, 5, 6, 7, 8]]
    
    input_df = pd.DataFrame(lst)
    
    input_df.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude']
    
    prediction = model.predict(input_df)
    return str(prediction)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
