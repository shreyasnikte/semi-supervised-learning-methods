# -*- coding: utf-8 -*-
"""
Author: Shreyas Nikte
"""
import numpy as np
import pandas as pd

from frameworks.CPLELearning import CPLELearningModel
from frameworks.SelfLearning import SelfLearningModel

import matplotlib.pyplot as plt

from scipy import stats
from scipy.sparse import csgraph
from scipy.optimize import minimize

from sklearn import preprocessing
from sklearn import svm
from sklearn import datasets, neighbors
from sklearn.semi_supervised import label_propagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from methods.scikitWQDA import WQDA



#Define the variables
n = [1, 10, 20, 40, 80, 160, 320, 640]
error_rate_knn = np.repeat(1.0,8)
error_rate_svm = np.repeat(1.0,8)
logLik_svm = np.repeat(0,8)
logLik_knn = np.repeat(0,8)
sd =1
n_neighbors = 10
#######################import the data into a matrix####################################

data = pd.read_csv('/home/shreyas/Documents/Lecture notes/Machine Learning/Semi-supervised/6649_0_magic04.txt', sep = ',', header = None)
label = data.loc[:,10]
del data[10]
label = label.replace('g', 0)
label = label.replace('h',1)
print label.shape
#normalize the data with stadard daviation = 1
normalized_X = preprocessing.scale(data)

#Randomly choose 25 data points with labels
X_labelled, X_other, y_labelled, y_other = train_test_split(data, label, train_size=25, random_state=42)

###############################################################################

#Create SVM classifier
lbl = "SVM (Supervised):"
print lbl
model = svm.SVC(gamma = 0.00001, C =100)
model.fit(X_labelled, y_labelled)
y_predict = model.predict(X_other)

accuracy = accuracy_score(y_other, y_predict)
error_rate = 1 - accuracy
logLik = -np.sum( stats.norm.logpdf(y_other, loc=y_predict, scale=sd) )
print 'SVM (Supervised) Error Rate:', error_rate, logLik
###############################################################################
###############################################################################


lbl = "KNN (Supervised):"
print lbl
model = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
model.fit(X_labelled, y_labelled)
y_predict = model.predict(X_other)

accuracy = accuracy_score(y_other, y_predict)
error_rate = 1 - accuracy
logLik = -np.sum( stats.norm.logpdf(y_other, loc=y_predict, scale=sd) )
print 'KNN (Supervised) Error Rate:', error_rate
###############################################################################



for i in range(0, 8):
    print '-------------------------------------------------------------'
    print 'Iteration: ', i
    print '-------------------------------------------------------------'
    #Randomly choose n values of unlabelled data
    X_unlabelled, X_extra, y_unlabelled, y_extra = train_test_split(X_other, y_other, train_size=n[i], random_state=42)
    y_minusone = np.repeat(-1,n[i])
    ################################################################################
    #Create training and target sets as per the training set distribution given above
    train_data = np.concatenate((X_labelled, X_unlabelled),axis =0)
    len_train = len(train_data)
    print 'No. of training data:', len_train
    #Final Traning labels
    train_labels = np.concatenate((y_labelled, y_minusone), axis = 0)
    len_labels = len(train_labels)
    print 'No. of training labels:', len_labels
    ##Print the number of test data
    print 'No. of test data:', len(y_extra)
    ################################################################################


    ################################################################################
    lbl = "CPLE(pessimistic) SVM:"
    print lbl
    model = CPLELearningModel(svm.SVC(kernel="rbf", probability=True), predict_from_probabilities=True, max_iter = 5000 )
    model.fit(train_data, train_labels)
    y_predict = model.predict(X_extra)

    accuracy = accuracy_score(y_extra, y_predict)
    print accuracy
    error_rate_svm[i] = 1 - accuracy
    logLik_svm[i] = -np.sum( stats.norm.logpdf(y_extra, loc=y_predict, scale=sd) )
    print 'CPLE Error Rate:', error_rate_svm[i], logLik_svm[i]
    ###############################################################################
    ################################################################################
    #Create the semi supervised KNN classifier
    lbl = "Label Propagation(KNN):"
    print lbl
    knn_model = label_propagation.LabelSpreading(kernel='knn', alpha=0.0001, max_iter=3000)
    knn_model.fit(train_data, train_labels)
    y_predict = knn_model.predict(X_extra)

    accuracy = accuracy_score(y_extra, y_predict)
    error_rate_knn[i] = 1 - accuracy
    logLik_knn[i] = -np.sum( stats.norm.logpdf(y_extra, loc=y_predict, scale=sd) )
    print 'KNN Error Rate:', error_rate_knn[i], logLik_knn[i]
    ################################################################################





    #EXTRA METHODS WHICH WERE USED FOR INITIAL EVALUATION PURPOSE
    ################################################################################
    #Create the semi supervised RBF (Radial basis function) classifier
    lbl = "Label Propagation(RBF):"
    print lbl
    rbf_model = label_propagation.LabelSpreading(kernel='rbf', gamma = 20,max_iter=50, tol=0.0001)
    rbf_model.fit(train_data, train_labels)
    y_predict = rbf_model.predict(X_extra)

    accuracy = accuracy_score(y_extra, y_predict)
    error_rate = 1 - accuracy
    logLik = -np.sum( stats.norm.logpdf(y_extra, loc=y_predict, scale=sd) )
    print 'RBF Error Rate:', error_rate, logLik
    ################################################################################
    ###############################################################################
    lbl = "Self Learning Model:"
    print lbl
    model = SelfLearningModel(WQDA())
    model.fit(train_data, train_labels)
    y_predict = model.predict(X_extra)
    # Calculate the negative log-likelihood as the negative sum of the log of a normal
    # PDF where the observed values are normally distributed around the mean (yPred)
    # with a standard deviation of sd

    accuracy = accuracy_score(y_extra, y_predict)
    error_rate = 1 - accuracy
    logLik = -np.sum( stats.norm.logpdf(y_extra, loc=y_predict, scale=sd) )
    print 'Self Learning Error Rate:', error_rate, logLik
    ###############################################################################
