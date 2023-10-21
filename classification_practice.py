#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:14:10 2023

"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# =============================================================================
# dataset can be downloaded from here:
# https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data
# =============================================================================

# save filepath to variable for easier access
file_path = '/Users/yubo/Downloads/archive/train.csv'
# test_file_path = '/Users/yubo/Downloads/archive/test.csv'

df = pd.read_csv(file_path)  # read the data and store data in DataFrame
# test_data = pd.read_csv(test_file_path)

# =============================================================================
# print a summary of the data, and check if there is any missing value,
# because many models cannot use samples with any missing values.
# =============================================================================

# not all columns will be displayed without this line of code
pd.set_option('display.max_columns', None)

print(df.describe())
# print(df.info)
# print(df.isnull().sum())
# print(val_data.describe())
# print(concat_data.describe())

x = df.drop('price_range', axis=1)  # x is feature matrix
y = df['price_range']  # y is label vector

# print(x.describe())
# print('---------')
# print(y.describe())

# =============================================================================
# feature engineering
# =============================================================================

scaler = StandardScaler()
X = scaler.fit_transform(x)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1)

# =============================================================================
# choose the best k for KNN
# =============================================================================
best_score = 0.0
best_k = 0
for k in range(1, 200):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    if score > best_score:
        best_score = score
        best_k = k
print("best k is: ", best_k, "best score is: ", best_score, '\n')

# =============================================================================
# val several models
# =============================================================================

# load package, if not, [error] missing 1 required positional argument: 'X' occurs.
KNN = KNeighborsClassifier(n_neighbors=best_k)
LR = LogisticRegression()
RF = RandomForestClassifier()
DT = DecisionTreeClassifier()
SVC = SVC()

KNN.fit(X_train, y_train)
LR.fit(X_train, y_train)
RF.fit(X_train, y_train)
DT.fit(X_train, y_train)
SVC.fit(X_train, y_train)

predict_KNN = KNN.predict(X_val)
predict_LR = LR.predict(X_val)
predict_RF = RF.predict(X_val)
predict_DT = DT.predict(X_val)
predict_SVC = SVC.predict(X_val)

cm_knn = confusion_matrix(predict_KNN, y_val)
cm_lr = confusion_matrix(predict_LR, y_val)
cm_rf = confusion_matrix(predict_RF, y_val)
cm_dt = confusion_matrix(predict_DT, y_val)
cm_svc = confusion_matrix(predict_SVC, y_val)

print('KNN:\n{}'.format(cm_knn))
print('Logistic Regression:\n{}'.format(cm_lr))
print('Random Forest Classifier:\n{}'.format(cm_rf))
print('Decision Tree Classifier:\n{}'.format(cm_dt))
print('SVC:\n{}'.format(cm_svc), '\n')

as_knn = accuracy_score(predict_KNN, y_val)*100
as_lr = accuracy_score(predict_LR, y_val)*100
as_rf = accuracy_score(predict_RF, y_val)*100
as_dt = accuracy_score(predict_DT, y_val)*100
as_svc = accuracy_score(predict_SVC, y_val)*100

print('KNN: {}%'.format(as_knn))
print('Logistic Regression: {}%'.format(as_lr))
print('Random Forest Classifier: {}%'.format(as_rf))
print('Decision Tree Classifier: {}%'.format(as_dt))
print('SVC: {}%'.format(as_svc))
