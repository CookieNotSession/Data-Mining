# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import jieba
import jieba.analyse
import pandas as pd
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
df=pd.read_excel("FDATA.xlsx")
tags=[]
bingo = 0
newgroup = []
T = []
T1 = []
T2 =[]
T3=[]
T4=[]
for i in range(df.shape[0]):
#for i in range(1):
    x = jieba.analyse.extract_tags(df["postContent"][i], 200)
    tags.append(' '.join(x))
    #print(",".join(tags))
    
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(tags))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()

kmeans = KMeans(n_clusters=5, random_state=0).fit(weight)
a = kmeans.labels_
print(a)


        

from sklearn import tree
from sklearn.model_selection import train_test_split

df['mainTag'] = pd.Categorical.from_array(df.mainTag).labels
X = weight
Y = df['mainTag']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
clf=tree.DecisionTreeClassifier(max_depth=25)
clf=clf.fit(X_train,Y_train)
mainTag = df['mainTag']

Tmost = 0
T1most = 0
T2most = 0
T3most = 0
T4most = 0
Tright = 0
T1right = 0
T2right = 0
T3right = 0
T4right = 0
for counting in range(len(a)):
    if(a[counting]==0):
        T.append(mainTag[counting])
        for counterT in range(5):
            if(Tmost<T.count(counterT)):
                Tmost = T.count(counterT)
                a[counting]=counterT
                for k in range(len(T)):
                    if(T[k]==counterT):
                        Tright+=1
                        break
    elif(a[counting]==1):
        T1.append(mainTag[counting])
        for counterT1 in range(5):
            if(T1most<T1.count(counterT1)):
                T1most = T1.count(counterT1)
                a[counting]=counterT1
                for k in range(len(T1)):
                    if(T1[k]==counterT1):
                        T1right+=1
                        break
    elif(a[counting]==2):
        T2.append(mainTag[counting])
        for counterT2 in range(5):
            if(T2most<T2.count(counterT2)):
                T2most = T2.count(counterT2)
                a[counting]=counterT2
                for k in range(len(T2)):
                    if(T2[k]==counterT2):
                        T2right+=1
                        break
    elif(a[counting]==3):
        T3.append(mainTag[counting])
        for counterT3 in range(5):
            if(T3most<T3.count(counterT3)):
                T3most = T3.count(counterT3)
                a[counting]=counterT3
                for k in range(len(T3)):
                    if(T3[k]==counterT3):
                        T3right+=1
                        break
    elif(a[counting]==4):
        T4.append(mainTag[counting])
        for counterT4 in range(5):
            if(T4most<T4.count(counterT4)):
                T4most = T4.count(counterT4)
                a[counting]=counterT4
                for k in range(len(T4)):
                    if(T4[k]==counterT4):
                        T4right+=1
                        break
    
                
print('科技Accuracy = '+ str(Tright/len(T)))
print('財經Accuracy = '+ str(T1right/len(T1)))
print('美食Accuracy = '+ str(T2right/len(T2)))
print('天氣Accuracy = '+ str(T3right/len(T3)))
print('運動Accuracy = '+ str(T4right/len(T4)))
print('Total Accuracy = '+ str((Tright+T1right+T2right+T3right+T4right)/(len(T)+len(T1)+len(T2)+len(T3)+len(T4))))



from sklearn.metrics import accuracy_score
Y_predict=clf.predict(X_test)
print("Decision Tree Accuracy=",accuracy_score(Y_test,Y_predict))

