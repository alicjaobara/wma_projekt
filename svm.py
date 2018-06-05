#!/usr/bin/env python

# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

def percent(cm):
    sump=0
    summ=0
    for x in range(0,2):
        sump = sump + cm[x,x]
        for y in range(0,2):
            summ = summ + cm[x,y]
    per=100*sump/summ
    return per

def svm(X,Y):
    # Splitting the dataset into the Training set and Test set (HOG)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf')
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm
    
# Importing the dataset (HOG)
datasetHOG = pd.read_csv('data/features/hog.txt', header = None, sep = ',')
X_hog = datasetHOG.iloc[:, :-1].values
Y_hog = datasetHOG.iloc[:, 288].values

# Importing the dataset (SIFT)
datasetSIFT = pd.read_csv('data/features/sift.txt', header = None, sep = ',')
X_sift = datasetHOG.iloc[:, :-1].values
Y_sift = datasetHOG.iloc[:, 70].values

cmHOG=svm(X_hog,Y_hog)
cmSIFT=svm(X_sift,Y_sift)
# Percentage HOG
percent(cmHOG)
percent(cmSIFT)

