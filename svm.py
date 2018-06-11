#!/usr/bin/env python

# Importing the libraries
import numpy as np
import pandas as pd

def percent(cm):
    sump=0
    summ=0
    for x in range(0,26):
        sump = sump + cm[x,x]
        for y in range(0,26):
            summ = summ + cm[x,y]
    per=100*sump/summ
    return per

def svmcm(X,Y):
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
X_sift = datasetSIFT.iloc[:, :-1].values
Y_sift = datasetSIFT.iloc[:, 288].values

# Importing the dataset (SURF)
datasetSURF = pd.read_csv('data/features/surf.txt', header = None, sep = ',')
X_surf = datasetSURF.iloc[:, :-1].values
Y_surf = datasetSURF.iloc[:, 288].values

# Dataset (SIFT SURF)
# datasetSIFTSURF=np.concatenate((datasetSIFT,datasetSURF),axis=1)
X_siftsurf = np.concatenate((X_sift,X_surf),axis=1)
Y_siftsurf = Y_sift

# Dataset (HOG SIFT)
#datasetHOGSIFT=np.concatenate((datasetHOG, datasetSIFT))
#X_hogsift = datasetHOGSIFT[:, :-1]
#Y_hogsift = datasetHOGSIFT[:, 288]
X_hogsift = np.concatenate((X_sift,X_hog),axis=1)
Y_hogsift = Y_hog

# Dataset (HOG SURF)
#datasetHOGSURF=np.concatenate((datasetHOG,datasetSURF))
#X_hogsurf = datasetHOGSURF[:, :-1]
#Y_hogsurf = datasetHOGSURF[:, 288]
X_hogsurf = np.concatenate((X_surf,X_hog),axis=1)
Y_hogsurf = Y_hog

# Dataset (HOG SIFT SURF)
#datasetHOGSIFTSURF=np.concatenate((datasetHOG,datasetSIFT,datasetSURF))
#X_hogsiftsurf = datasetHOGSIFTSURF[:, :-1]
#Y_hogsiftsurf = datasetHOGSIFTSURF[:, 288]
X_hogsiftsurf = np.concatenate((X_surf,X_hog,X_sift),axis=1)
Y_hogsiftsurf = Y_hog

# Making the Confusion Matrix
cmHOG=svmcm(X_hog,Y_hog)
cmSIFT=svmcm(X_sift,Y_sift)
cmSURF=svmcm(X_surf,Y_surf)
cmSIFTSURF=svmcm(X_siftsurf,Y_siftsurf)
cmHOGSIFT=svmcm(X_hogsift,Y_hogsift)
cmHOGSURF=svmcm(X_hogsurf,Y_hogsurf)
cmHOGSIFTSURF=svmcm(X_hogsiftsurf,Y_hogsiftsurf)

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