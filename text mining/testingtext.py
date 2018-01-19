#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:31:02 2017

@author: cookiehiker
"""

import jieba
import sys

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        corpus.append(" ".join(jieba.cut(line.split(',')[0], cut_all=False)))

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
print(tfidf.shape)

words = vectorizer.get_feature_names()
for i in range(len(tags)):
    print('----Document %d----' % (i))
    for j in range(len(words)):
        if tfidf[i,j] > 1e-5:
              print(words[j].encode('utf-8'), tfidf[i,j])