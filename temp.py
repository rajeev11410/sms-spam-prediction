# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
                               
# loading the data from csv file to a pandas Dataframe
raw_sms_data = pd.read_csv('C:/Users/HP/Desktop/college/sms_data.csv')
# replace the null values with a null string
sms_data = raw_sms_data.where((pd.notnull(raw_sms_data)),'')
# printing the first 5 rows of the dataframe
sms_data.head()
# checking the number of rows and columns in the dataframe
sms_data.shape
# label spam mail as 0;  ham mail as 1;

sms_data.loc[sms_data['Category'] == 'spam', 'Category',] = 0
sms_data.loc[sms_data['Category'] == 'ham', 'Category',] = 1
# separating the data as texts and label

X = sms_data['Message']

Y = sms_data['Category']# separating the data as texts and label

X = sms_data['Message']

Y = sms_data['Category']
#Splitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)

# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#print(X_train)
#print(X_train_features)
model = LogisticRegression()
# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

#print('Accuracy on training data : ', accuracy_on_training_data)
input_sms = ["click thiis link and get 2000 https://wxyz.com//"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_sms)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)
if (prediction[0]==1):
  print('ham sms')       
else:
  print('Spam sms')
