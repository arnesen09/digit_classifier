# -*- coding: utf-8 -*-
"""
Created on Fri May  5 08:12:31 2017

@author: Dane Arnesen

Kaggle 'Digit Recognizer' competition
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from sklearn.metrics import confusion_matrix
from keras.regularizers import l1

#==============================================================#
#              Helper Functions                                #  
#==============================================================#

# Mapping array of dummies back to the actual digit
def get_numbers(vector):
    numbers = []
    for v in vector:
        ind = np.argmax(v)
        numbers.append(ind)
    return numbers

#==============================================================#
#              Exploration & Preprocessing                     #  
#==============================================================#

# Importing the raw data
df = pd.read_csv('C://Users/g557428/Desktop/train.csv')

# Viewing the dimensions
print(df.shape)
print(df.dtypes)

# Makeup of the target attributes - they are pretty evenly split.
df.groupby('label')['label'].count()

# Counting null values for each column
print(df.isnull().sum())

# Viewing summary statistics of each column
print(df.describe().transpose())

# Creating dummies for the target
df['zero'] = df['label'].apply(lambda x : 1 if x == 0 else 0)
df['one'] = df['label'].apply(lambda x : 1 if x == 1 else 0)
df['two'] = df['label'].apply(lambda x : 1 if x == 2 else 0)
df['three'] = df['label'].apply(lambda x : 1 if x == 3 else 0)
df['four'] = df['label'].apply(lambda x : 1 if x == 4 else 0)
df['five'] = df['label'].apply(lambda x : 1 if x == 5 else 0)
df['six'] = df['label'].apply(lambda x : 1 if x == 6 else 0)
df['seven'] = df['label'].apply(lambda x : 1 if x == 7 else 0)
df['eight'] = df['label'].apply(lambda x : 1 if x == 8 else 0)
df['nine'] = df['label'].apply(lambda x : 1 if x == 9 else 0)

# Splitting the dataframe into x and y
x = np.array(df)[:,1:785]
y = np.array(df)[:,785:796]

# Normalizing the x values
x = minmax_scale(x)

# Splitting into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=19861005)

#==============================================================#
#    Basic Fully Connected Neural Network                      #
#   677th of 1641 place public leaderboard                     #  
#==============================================================#

# FC model
fit = Sequential()
fit.add(Dense(600, input_dim=784, activation='relu'))
fit.add(Dense(500, activation='relu'))
fit.add(Dense(400, activation='relu'))
fit.add(Dense(300, activation='relu'))
fit.add(Dense(200, activation='relu'))
fit.add(Dense(100, activation='relu'))
fit.add(Dense(50, activation='relu'))
fit.add(Dense(25, activation='relu'))
fit.add(Dense(10, activation='softmax'))

# Compile the model. Specify loss function, optimizer, and metrics
fit.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Fitting the model. Specifying number of epochs and the batch size
fit.fit(x_train, y_train, epochs=500, batch_size=1000)

# Make the predictions on the train dataset
pred = fit.predict(x_test)

# Convert final predictions to numbers
final_pred = get_numbers(pred)

# Convert y vector into numbers
actual = get_numbers(y_test)
    
# Display the confusion matrix
cm = confusion_matrix(actual, final_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
acc = np.sum(np.diagonal(cm))/np.sum(cm)

# serialize model to JSON
fit_json = fit.to_json()
with open('C://Users/g557428/Desktop/digit_classifier_1.json', 'w') as json_file:
    json_file.write(fit_json)
# serialize weights to HDF5
fit.save_weights('C://Users/g557428/Desktop/digit_classifier_1.h5')
print("Saved model to disk")

#==============================================================#
#    Convolutional NN                                          #        
#==============================================================#


