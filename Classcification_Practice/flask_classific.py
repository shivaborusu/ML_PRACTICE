#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 01:59:10 2020

@author: shivaborusu
"""

#unpacking pickle file for flaskdeployment

import pickle as pkl
from flask import Flask, request, make_response, send_file
from flasgger import Swagger
import pandas as pd

with open("class.pkl", "rb") as pkl_file:
    model = pkl.load(pkl_file)
    
app = Flask(__name__)
Swagger(app)    

@app.route('/predict', methods=['POST'])
def predict_class():
    """Example file endpoint returning a prediction
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    
    responses:
      500:
        description: Fail !!!
      200:
        description: Success !!! Result is saved       
    """
    #request.files.get("input_file")
    csv_file = request.files.get("input_file")
    data1 = pd.read_csv(csv_file, index_col = None)
    result = model.predict(data1)
    result = pd.DataFrame(result.reshape(result.size,1), columns= ['Target'])
    combine_result = pd.concat([data1,result], axis=1)
    combine_result.to_csv("./result.csv")
    response = make_response(send_file("./result.csv", 
                             attachment_filename='result.csv',
                             as_attachment=True))
    return response

if __name__ == "__main__":
    app.run("127.0.0.1", port=5000)
