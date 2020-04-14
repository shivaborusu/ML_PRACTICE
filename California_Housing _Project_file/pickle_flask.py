#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:41:59 2020

@author: shivaborusu
"""

#california data set unpickling and flask
import pickle as pkl
from flask import Flask, request, send_file
import pandas as pd
from flasgger import Swagger
#from werkzeug.utils import secure_filename

infile = open("./pkl_file.pkl","rb")

model = pkl.load(infile)

infile.close()

app = Flask(__name__)
Swagger(app)


@app.route('/predict_file', methods=["POST"])
def predict_file():
    """Example file endpoint returning a prediction
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"))
    prediction = model.predict(input_data)
    prediction = pd.DataFrame(prediction.reshape(prediction.size,1))
    prediction.to_csv("./predictions.csv")
    return "Success"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
