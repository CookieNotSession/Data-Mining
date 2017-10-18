#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:01:36 2017

@author: cookie040667
"""

# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("nycrollingsales.csv",encoding = "ISO-8859-1") #UnicodeDecodeError做處理

df = df.drop(df.columns[0],axis=1) #A
df = df.drop('NEIGHBORHOOD',axis=1) #C 
df = df.drop('TAX CLASS AT PRESENT',axis=1) #E
df = df.drop('BLOCK',axis=1) #F
df = df.drop('LOT',axis=1) #G
df = df.drop('EASE-MENT',axis=1) #H
df = df.drop('ADDRESS',axis=1) #J
df = df.drop('APARTMENT NUMBER',axis=1)#K
df = df.drop('COMMERCIAL UNITS',axis=1)#N
df = df.drop('SALE DATE',axis=1)#V
df = df.drop('BUILDING CLASS AT TIME OF SALE',axis=1)

df = df.drop('RESIDENTIAL UNITS',axis=1)



df = df.drop('BOROUGH',axis=1)
#df = df.drop('TAX CLASS AT TIME OF SALE',axis=1)
df = df.drop('GROSS SQUARE FEET',axis=1)
#df = df.drop('LAND SQUARE FEET',axis=1)
#df = df.drop('BUILDING CLASS CATEGORY',axis=1)




df = df[df['SALE PRICE'].str.contains('- ')==False]
#df = df[df['GROSS SQUARE FEET'].str.contains('- ')==False]
df = df[df['LAND SQUARE FEET'].str.contains('- ')==False] #將欄位中有dash的不顯示
df['SALE PRICE'] = df['SALE PRICE'].apply(int)


df = df.drop(df[df['SALE PRICE'] ==1].index)
df = df.drop(df[df['SALE PRICE'] ==10].index)
df = df.drop(df[df['SALE PRICE'] ==0].index)
Q1 = np.percentile(df['SALE PRICE'], 10)
Q3 = np.percentile(df['SALE PRICE'], 90)
df = df.drop(df[df['SALE PRICE'] >Q3].index)
df = df.drop(df[df['SALE PRICE'] <Q1].index)
target=df['SALE PRICE']


#Q3 = target.quantile(0.75)
#groups = df.groupby('SALE PRICE')
#q3 = groups.quantile(q=0.75)




#df = pd.get_dummies(df,columns=['NEIGHBORHOOD'])
#df = pd.get_dummies(df,columns=['BUILDING CLASS AT TIME OF SALE'])
df = pd.get_dummies(df,columns=['BUILDING CLASS AT PRESENT'])
df = pd.get_dummies(df,columns=['BUILDING CLASS CATEGORY'])
#df = pd.get_dummies(df,columns=['LOT'])
#df = pd.get_dummies(df,columns=['ZIP CODE'])
#df = pd.get_dummies(df,columns=['TOTAL UNITS'])
#df = pd.get_dummies(df,columns=['LAND SQUARE FEET'])
#df = pd.get_dummies(df,columns=['GROSS SQUARE FEET'])
#df = pd.get_dummies(df,columns=['YEAR BUILT'])
#df = pd.get_dummies(df,columns=['TAX CLASS AT TIME OF SALE']) #key!不要轉dummy

train=df.drop('SALE PRICE',axis=1)


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
