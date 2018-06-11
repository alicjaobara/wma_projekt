#!/usr/bin/env python

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def percent(cm):
    sump = 0
    summ = 0
    for x in range(0,26):
        sump = sump + cm[x,x]
        for y in range(0,26):
            summ = summ + cm[x,y]
    per = 100*sump/summ
    return per

def svmcm(X,Y,Xtest,Ytest):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    Xtest = sc.transform(Xtest)
    # Fitting SVM to the Training set
    classifier = SVC(kernel = 'rbf')
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred2 = classifier.predict(Xtest)
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    #cm2 = confusion_matrix(Ytest, y_pred2)
    return cm, y_pred2
    
# Importing the dataset (HOG)
datasetHOG = pd.read_csv('data/features/hog.txt', header = None, sep = ',')
X_hog = datasetHOG.iloc[:, :-1].values
Y_hog = datasetHOG.iloc[:, 288].values

datasetHOG2 = pd.read_csv('data/features/hog10.txt', header = None, sep = ',')
X_hogtest = datasetHOG2.iloc[:, :-1].values
Y_hogtest = datasetHOG2.iloc[:, 288].values

# Importing the dataset (SIFT)
datasetSIFT = pd.read_csv('data/features/sift.txt', header = None, sep = ',')
X_sift = datasetSIFT.iloc[:, :-1].values
Y_sift = datasetSIFT.iloc[:, 288].values

datasetSIFT2 = pd.read_csv('data/features/sift10.txt', header = None, sep = ',')
X_sifttest = datasetSIFT2.iloc[:, :-1].values
Y_sifttest = datasetSIFT2.iloc[:, 288].values

# Importing the dataset (SURF)
datasetSURF = pd.read_csv('data/features/surf.txt', header = None, sep = ',')
X_surf = datasetSURF.iloc[:, :-1].values
Y_surf = datasetSURF.iloc[:, 288].values

datasetSURF2 = pd.read_csv('data/features/surf10.txt', header = None, sep = ',')
X_surftest = datasetSURF2.iloc[:, :-1].values
Y_surftest = datasetSURF2.iloc[:, 288].values

# Dataset (SIFT SURF)
X_siftsurf = np.concatenate((X_sift,X_surf),axis=1)
Y_siftsurf = Y_sift
X_siftsurftest = np.concatenate((X_sifttest,X_surftest),axis=1)
Y_siftsurftest = Y_sifttest

# Dataset (HOG SIFT)
X_hogsift = np.concatenate((X_sift,X_hog),axis=1)
Y_hogsift = Y_hog
X_hogsifttest = np.concatenate((X_sifttest,X_hogtest),axis=1)
Y_hogsifttest = Y_hogtest

# Dataset (HOG SURF)
X_hogsurf = np.concatenate((X_surf,X_hog),axis=1)
Y_hogsurf = Y_hog
X_hogsurftest = np.concatenate((X_surftest,X_hogtest),axis=1)
Y_hogsurftest = Y_hogtest

# Dataset (HOG SIFT SURF)
X_hogsiftsurf = np.concatenate((X_surf,X_hog,X_sift),axis=1)
Y_hogsiftsurf = Y_hog
X_hogsiftsurftest = np.concatenate((X_surftest,X_hogtest,X_sifttest),axis=1)
Y_hogsiftsurftest = Y_hogtest

# SVM
cmHOG, y_predHOG = svmcm(X_hog,Y_hog,X_hogtest,Y_hogtest)
cmSIFT, y_predHSIFT = svmcm(X_sift,Y_sift)
cmSURF, y_predSURF = svmcm(X_surf,Y_surf)
cmSIFTSURF, y_predSIFTSURF = svmcm(X_siftsurf,Y_siftsurf)
cmHOGSIFT, y_predHOGSIFT = svmcm(X_hogsift,Y_hogsift)
cmHOGSURF, y_predHOGSURF = svmcm(X_hogsurf,Y_hogsurf)
cmHOGSIFTSURF, y_predHOGSIFTSURF = svmcm(X_hogsiftsurf,Y_hogsiftsurf)

file = open('data/results/test500_25.txt','w') 

# Percentage HOG
file.write("HOG\t")
p=percent(cmHOG)
file.write('%s\n' % p)

# Percentage SIFT
file.write("SIFT\t")
p=percent(cmSIFT)
file.write('%s\n' % p)

# Percentage SURF
file.write("SURF\t")
p=percent(cmSURF)
file.write('%s\n' % p)

# Percentage SIFT SURF
file.write("SIFT SURF\t")
p=percent(cmSIFTSURF)
file.write('%s\n' % p)

# Percentage HOG SIFT
file.write("HOG SIFT\t")
p=percent(cmHOGSIFT)
file.write('%s\n' % p)

# Percentage HOG SURF
file.write("HOG SURF\t")
p=percent(cmHOGSURF)
file.write('%s\n' % p)

# Percentage HOG SIFT SURF
file.write("HOG SIFT SUR\t ")
p=percent(cmHOGSIFTSURF)
file.write('%s\n' % p)

file.close() 