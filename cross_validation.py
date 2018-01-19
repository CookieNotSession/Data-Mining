#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:13:24 2017

@author: cookie040667
"""

# imports
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import ensemble

df=pd.read_csv("data.csv") 
df = pd.get_dummies(df,columns=['workclass'])
df = pd.get_dummies(df,columns=['education'])
df = pd.get_dummies(df,columns=['marital_status'])
df = pd.get_dummies(df,columns=['occupation'])
df = pd.get_dummies(df,columns=['relationship'])
df = pd.get_dummies(df,columns=['race'])
df = pd.get_dummies(df,columns=['sex'])
df = pd.get_dummies(df,columns=['native_country'])
df['income'] = np.where(df.income.isin([' <=50K']),'0', df['income'])
df['income'] = np.where(df.income.isin([' >50K']),'1', df['income'])

data = df.drop('income',axis=1)
target = df.income

def K_fold_CV(k,data,target):
    subset_size = df.shape[0]//k
    Accuracy = 0
    index = 0
    for i in range(k):
        Y_testing = target[index:index+subset_size+1]
        X_testing = data[index:index+subset_size+1]
        Y_training = np.concatenate((target[0:index],target[subset_size+index+1:]),axis=0)
        X_training = np.concatenate((data[0:index],data[subset_size+index+1:]),axis=0)
        index = index + subset_size 
        
        clf = ensemble.GradientBoostingClassifier()
        clf.fit(X_training,Y_training)
        
        Y_predict=clf.predict(X_testing)
        single_accuracy = accuracy_score(Y_testing,Y_predict)
        Accuracy = Accuracy + single_accuracy
    Accuracy = Accuracy / k
    return Accuracy

Answer = K_fold_CV(10,data,target)
print(Answer)
#print(Answer)

        
"""       
        
#df = df.drop('BUILDING CLASS AT TIME OF SALE',axis=1)

#df = df[df['GROSS SQUARE FEET'].str.contains('- ')==False]

#target=df['SALE PRICE']
#df = pd.get_dummies(df,columns=['BUILDING CLASS AT PRESENT'])
#train=df.drop('SALE PRICE',axis=1)


X_train,X_test,Y_train,Y_test=train_test_split(train,target,test_size=0.25)


# Split the targets into training/testing sets
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

print(metrics.mean_absolute_error(Y_test, Y_pred))


forest = RandomForestRegressor(max_depth=2, random_state=0)
forest.fit(X_train, Y_train)
Y_pred = forest.predict(X_test)
print(metrics.mean_absolute_error(Y_test, Y_pred))
"""