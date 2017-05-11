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
from keras.layers import Dense, Input, Dropout, Flatten
from keras.models import Model, Sequential
from keras.regularizers import l1
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from sklearn.metrics import confusion_matrix

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
df = pd.read_csv('C://Users/g557428/Projects/digit_classifier/data/raw/train.csv')

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=19880124)

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
#    Autoencoder + Deep NN                                     # 
#==============================================================#

# First encoder / decoder layer
inputs = Input(shape=(784,))
encoder = Dense(100, activation='sigmoid')(inputs)
decoder = Dense(784, activation='sigmoid')(encoder)
autoencoder = Model(inputs, decoder)
autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=1000)

# Get the weights from the encoder layer
w1 = autoencoder.layers[1].get_weights()
               
# Get the x_hat values to feed into the next model
fit = Model(inputs, encoder)
x_hat = fit.predict(x_train)

# Second encoder / decoder layer
inputs = Input(shape=(100,))
encoder = Dense(50, activation='sigmoid')(inputs)
decoder = Dense(100, activation='sigmoid')(encoder)
autoencoder = Model(inputs, decoder)
autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop')
autoencoder.fit(x_hat, x_hat, epochs=50, batch_size=100)

# Get the weights from the encoder layer
w2 = autoencoder.layers[1].get_weights()
               
# Get the x_hat values to feed into the next model
fit = Model(inputs, encoder)
x_hat = fit.predict(x_hat)

# Third encoder / decoder layer
inputs = Input(shape=(50,))
encoder = Dense(25, activation='relu')(inputs)
decoder = Dense(50, activation='relu')(encoder)
autoencoder = Model(inputs, decoder)
autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop')
autoencoder.fit(x_hat, x_hat, epochs=50, batch_size=100)

# Get the weights from the encoder layer
w3 = autoencoder.layers[1].get_weights()
               
# Get the x_hat values to feed into the next model
fit = Model(inputs, encoder)
x_hat = fit.predict(x_hat)

# Final output layer
inputs = Input(shape=(25,))
encoder = Dense(10, activation='softmax')(inputs)
autoencoder = Model(inputs, encoder)
autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
autoencoder.fit(x_hat, y_train, epochs=25, batch_size=100)

# Get weights from the final output layer
w4 = autoencoder.layers[1].get_weights()

# Now train a deep neural network using pre-trained weights and original x_train
fit = Sequential()
fit.add(Dense(100, input_dim=784, activation='relu', weights=w1))
fit.add(Dense(50, activation='relu', weights=w2))
fit.add(Dense(25, activation='relu'))
fit.add(Dense(10, activation='softmax', weights=w4))
fit.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
fit.fit(x_train, y_train, epochs=50, batch_size=100)

# Make the predictions on the train dataset
pred = fit.predict(x_test)

# Convert final predictions to numbers
final_pred = get_numbers(pred)

# Convert y vector into numbers
actual = get_numbers(y_test)
    
# Display the confusion matrix
cm = confusion_matrix(actual, final_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
acc = np.sum(np.diagonal(cm))/np.sum(cm)

#==============================================================#
#    Simple Convolutional NN                                   # 
#    Two convolutional layers followed by pooling followed     #
#    followed by FC layers for prediction                      #
#==============================================================#

# Reshape the test and train dataset
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Creating the model
fit = Sequential()

fit.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu', padding='same', activity_regularizer=l1(10e-5)))
fit.add(MaxPooling2D(pool_size=(2,2)))
fit.add(Dropout(0.25))

fit.add(Conv2D(64, (3,3), activation='relu', padding='same', activity_regularizer=l1(10e-5)))
fit.add(MaxPooling2D(pool_size=(2,2)))
fit.add(Dropout(0.25))

fit.add(Conv2D(128, (3,3), activation='relu', padding='same', activity_regularizer=l1(10e-5)))
fit.add(MaxPooling2D(pool_size=(2,2)))
fit.add(Dropout(0.25))

fit.add(Flatten())
fit.add(Dense(64, activation='relu'))
fit.add(Dropout(0.5))
fit.add(Dense(32, activation='relu'))
fit.add(Dropout(0.5))
fit.add(Dense(16, activation='relu'))
fit.add(Dropout(0.5))
fit.add(Dense(10, activation='softmax'))

# Compile the model. Specify loss function, optimizer, and metrics
fit.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Fitting the model. Specifying number of epochs and the batch size
fit.fit(x_train, y_train, epochs=10, batch_size=5000)

# Make the predictions on the train dataset
pred = fit.predict(x_test)

# Convert final predictions to numbers
final_pred = get_numbers(pred)

# Convert y vector into numbers
actual = get_numbers(y_test)
    
# Display the confusion matrix
cm = confusion_matrix(actual, final_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
acc = np.sum(np.diagonal(cm))/np.sum(cm)

