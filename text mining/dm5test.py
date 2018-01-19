#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 11:23:31 2017

@author: cookiehiker
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
for i in range(df.shape[0]):
    x = jieba.analyse.extract_tags(df["postContent"][i], 50)
    tags.append(' '.join(x))

    