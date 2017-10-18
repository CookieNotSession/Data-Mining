#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:47:31 2017

@author: cookie040667
"""
import pandas as pd

df=pd.read_csv("desktop/character-deaths.csv")
#load csv datasets in
df=df.fillna(0) #change null column into 0
#df_load.loc[df_load['Death Year']]>0:
#df_load.loc[df_load['Death Year']>0,'Death Year']=1
#df['Death Year'][df['Death Year']!=0]=1
df.loc[df['Death Year'] > 0, 'Death Year'] = 1.0
df = pd.get_dummies(df,columns=['Allegiances'])
df = df.drop('Name',axis=1)
df = df.drop('Book of Death',axis=1)
df = df.drop('Death Chapter',axis=1)

from sklearn import tree
from sklearn.model_selection import train_test_split
X=df.drop('Death Year',axis=1)
Y=df['Death Year']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
clf=tree.DecisionTreeClassifier(max_depth=4)
clf=clf.fit(X_train,Y_train)
import graphviz
dot_data=tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph.render("df")

dot_data = tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
graph

Y_predict=clf.predict(X_test)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
print("Precision Rate=",precision_score(Y_test,Y_predict))
print("Recall Rate=",recall_score(Y_test,Y_predict))
print("Accuracy=",accuracy_score(Y_test,Y_predict))
