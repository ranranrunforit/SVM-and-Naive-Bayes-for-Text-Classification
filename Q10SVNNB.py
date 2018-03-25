# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 18:30:09 2018

@author: chaoran
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np


# NaiveBayes algorithm
def NB(categories):
    #get the data from 4 folder
    all_four = fetch_20newsgroups(subset='all',categories=categories, shuffle=True, random_state=42)
    #split the data to training data and test data    
    train_data, test_data, train_category, test_category = train_test_split(all_four.data, all_four.target, test_size = 0.3)   
    #unigram
    #pipeline for text feature extraction and evaluation    
    #tokenizer => transformer => MultinomialNB classifier
    text_clf = Pipeline([('vect', CountVectorizer( analyzer = 'word',ngram_range = (1,1), stop_words='english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=.01))])
    text_clf = text_clf.fit(train_data, train_category)
    #evaluate on test set
    predicted = text_clf.predict(test_data)
    print("************************* Naive Bayes Model *************************")
    print("Newsgroup Categories : ", categories )
    print("Unigram Accuracy : {}% \n".format(np.mean(predicted == test_category)*100))
    print(metrics.classification_report(test_category, predicted, target_names=categories))
    print("Unigram Confusion Matrix : \n", metrics.confusion_matrix(test_category, predicted))
    print("\n")
    #bigram
    #pipeline for text feature extraction and evaluation    
    #tokenizer => transformer => MultinomialNB classifier    
    bigram_clf = Pipeline([('vect', CountVectorizer( analyzer = 'word',ngram_range = (2,2), stop_words='english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=.01))])
    bigram_clf = bigram_clf.fit(train_data, train_category) 
    #evaluate on test set    
    predicted = bigram_clf.predict(test_data)
    print("Bigram Accuracy : {}% \n".format(np.mean(predicted == test_category)*100))
    print(metrics.classification_report(test_category, predicted, target_names=categories))
    print("Bigram Confusion Matrix : \n", metrics.confusion_matrix(test_category, predicted)) 
    
    
# SVM algorithm
def SVM(categories):
    #get the data from 4 folder
    all_four = fetch_20newsgroups(subset='all',categories=categories, shuffle=True, random_state=42)
    #split the data to training data and test data     
    train_data, test_data, train_category, test_category = train_test_split(all_four.data, all_four.target, test_size = 0.3)
    #unigram
    #pipeline for text feature extraction and evaluation    
    #tokenizer => transformer => MultinomialNB classifier
    #LinearSVC()
    unigram_clf = Pipeline([('vect', CountVectorizer( analyzer = 'word',ngram_range = (1,1), stop_words='english')),('tfidf', TfidfTransformer()),('clf', LinearSVC())])
    unigram_clf = unigram_clf.fit(train_data, train_category) 
    #evaluate on test set    
    predicted = unigram_clf.predict(test_data)
    print("************************* SVM Model *************************")
    print("Newsgroup Categories : ", categories )
    print("Unigram Accuracy : {}% \n".format(np.mean(predicted == test_category)*100))
    print(metrics.classification_report(test_category, predicted, target_names=categories))
    print("Unigram Confusion Matrix : \n", metrics.confusion_matrix(test_category, predicted))
    print("\n")
    #bigram   
    #pipeline for text feature extraction and evaluation    
    #tokenizer => transformer => MultinomialNB classifier
    #SGDClassifier(loss='hinge')
    bigram_clf = Pipeline([('vect', CountVectorizer( analyzer = 'word',ngram_range = (2,2), stop_words='english')),('tfidf', TfidfTransformer()),('clf', LinearSVC())])
    bigram_clf = bigram_clf.fit(train_data, train_category) 
    #evaluate on test set    
    predicted = bigram_clf.predict(test_data)
    print("Bigram Accuracy : {}% \n".format(np.mean(predicted == test_category)*100))
    print(metrics.classification_report(test_category, predicted, target_names=categories))
    print("Bigram Confusion Matrix : \n", metrics.confusion_matrix(test_category, predicted))    


    
# selected categories
categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']
NB(categories)
SVM(categories)
