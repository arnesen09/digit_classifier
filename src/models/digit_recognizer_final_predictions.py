# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:49:59 2017

@author: G557428
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json

#==============================================================#
#                Preprocessing                                 #  
#==============================================================#

# Importing the raw data
df = pd.read_csv('C://Users/g557428/Desktop/test.csv')

# Normalizing the x values
x = minmax_scale(np.array(df))

#==============================================================#
#                User defined functions                        #  
#==============================================================#

def get_numbers(vector):
    numbers = []
    for v in vector:
        ind = np.argmax(v)
        numbers.append(ind)
    return numbers

#==============================================================#
#                Model 1                                       #  
#==============================================================#

# load json and create model
json_file = open('C://Users/g557428/Desktop/digit_classifier_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
fit = model_from_json(loaded_model_json)
# load weights into new model
fit.load_weights('C://Users/g557428/Desktop/digit_classifier_1.h5')
print("Loaded model from disk")

# Make the predictions on the train dataset
pred = fit.predict(x)
pred = get_numbers(pred)

# Converting predictions to dataframe
pred = pd.DataFrame(pred, columns=['Label'])
pred.reset_index(inplace=True)
pred['index'] = pred['index'] + 1
pred.rename(columns={'index': 'ImageId'}, inplace=True)

# Viewing the head  
pred.head()

# Export to csv
pred.to_csv('C://Users/g557428/Desktop/predictions.csv', index=False)