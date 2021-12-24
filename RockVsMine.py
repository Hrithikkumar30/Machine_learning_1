# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 08:18:42 2021

@author: alfa
"""
# importing the libraries which will be used in this project

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  #Used to find the accuracy of the model

# Data Collection and Data Proceesing

sonar_data = pd.read_csv('C:/Users/alfa/Desktop/machineLearning/rockVsmineProject/Copy of sonar data.csv',header=None) # it will read our csv file
#print(sonar_data.head())  # THis will display 1st 5 rows of our data sets.


#print(sonar_data.shape)#checking number of rows and column
#print(sonar_data.describe()) # GIves all the statical value of the data

#print(sonar_data[60].value_counts()) # it tells how many provided data of rock and mine seperately 60 because at the specification of Rock and Mine is done on column no. 60th
"""
M--> Mine
R--> Rock
"""
grouped_mean = sonar_data.groupby(60).mean()
#print(grouped_mean)

#Seperating Data and labels

X = sonar_data.drop(columns = 60, axis =1)
Y = sonar_data[60]
#print(X)
#print(Y)

#Training and Test Data

X_train , X_test , Y_train, Y_test =  train_test_split(X, Y, test_size=0.1, stratify = Y, random_state=1)
#print(X.shape , X_train.shape , X_test.shape)


#MODEL Training --> Logistlic Regression

#training logistic regression model with training data 

model = LogisticRegression()
model.fit(X_train, Y_train)

#print(X_train)
#print(Y_train)


#Model Evaluation

#accuracy on trainig data
X_train_prediction = model.predict(X_train)
trainig_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("accuracy of training data is: ", trainig_data_accuracy)

#accuracy on testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("accuracy of test data: " ,test_data_accuracy)

#Making a Predictive system
input_data = (0.0201,0.0178,0.0274,0.0232,0.0724,0.0833,0.1232,0.1298,0.2085,0.2720,0.2188,0.3037,0.2959,0.2059,0.0906,0.1610,0.1800,0.2180,0.2026,0.1506,0.0521,0.2143,0.4333,0.5943,0.6926,0.7576,0.8787,0.9060,0.8528,0.9087,0.9657,0.9306,0.7774,0.6643,0.6604,0.6884,0.6938,0.5932,0.5774,0.6223,0.5841,0.4527,0.4911,0.5762,0.5013,0.4042,0.3123,0.2232,0.1085,0.0414,0.0253,0.0131,0.0049,0.0104,0.0102,0.0092,0.0083,0.0020,0.0048,0.0036)
input_data_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=="R"):
    print("Rock is Underneath")
else:
    print("Danger!! It is a mine")



