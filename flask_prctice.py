#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:09:36 2020

@author: shivaborusu
"""

from flask import Flask

app = Flask (__name__)

@app.route('/')
def my_first_flask_app():
    print("Hello World")
    return "Hello World, this is in return statement"
    
if __name__ == '__main__':
    app.run()