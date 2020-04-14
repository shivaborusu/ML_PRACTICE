#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:46:03 2019

@author: shivaborusu
"""
import pandas as pd

import matplotlib.pyplot as plt

df=pd.DataFrame([[1, "Shiva", 6], [2, "Murthy", 7], [3, "Venkat", 12]], columns=['x','y','z'])


df.plot(kind='scatter', x=df['x'], y=df['z'])