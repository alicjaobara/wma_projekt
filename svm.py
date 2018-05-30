#!/usr/bin/env python

# Importing the libraries
# math
#import numpy as np
# plot
# import matplotlib.pyplot as plt
# datasets managing
import pandas as pd

# Importing the dataset (HOG)
datasetHOG = pd.read_csv('data/hogFeatures/hog.txt', header = None, sep = ',')
X_hog = datasetHOG.iloc[:, :-1].values
Y_hog = datasetHOG.iloc[:, 288].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_hog, Y_hog, test_size = 0.25)

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

# Percentage
sump=0
summ=0
for x in range(0,26):
    sump = sump + cm[x,x]
    for y in range(0,26):
        summ = summ + cm[x,y]
    
percent=100*sump/summ
print(percent)

