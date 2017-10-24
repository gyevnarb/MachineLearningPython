# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:01:49 2017

@author: Bálint
"""
import os
import pandas as pd
import numpy.random as rnd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_union, make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV

TITANIC_PATH = 'datasets/titanic'

def load_training_data(dataset=TITANIC_PATH):
    train_path = os.path.join(dataset, 'train.csv')
    test_path = os.path.join(dataset, 'test.csv')
    #test_survival_path = os.path.join(dataset, 'gender_submission.csv')
    return pd.read_csv(train_path), pd.read_csv(test_path)#, pd.read_csv(test_survival_path)

def print_statistics(model, X_train, y_train):
    some_index = rnd.randint(0, len(X_train))
    some_person = X_train[some_index].reshape(1, -1)
    print('Prediction: %s True: [%s]' % (model.predict(some_person), y_train[some_index]))
    y_train_predict = cross_val_predict(model, X_train, y_train, cv=3)
    print('Accuracy:', cross_val_score(model, X_train, y_train, cv=3))
    print('Precision:', precision_score(y_train, y_train_predict))
    print('Recall:', recall_score(y_train, y_train_predict))
    print('F1 Score:', f1_score(y_train, y_train_predict))

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class MultipleLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.get_dummies(X[self.attribute_names])

train, test = load_training_data()
y_train = train['Survived']

num_attribs   = ['Age', 'SibSp', 'Parch', 'Fare']
label_attribs = ['Pclass']
cat_attribs   = ['Sex', 'Embarked']

num_pipeline = make_pipeline(DataFrameSelector(num_attribs), 
                             Imputer(strategy='median'), StandardScaler())
    
cat_pipeline = make_union(
            make_pipeline(MultipleLabelBinarizer(cat_attribs)),
            make_pipeline(DataFrameSelector(label_attribs), LabelBinarizer()),
        )

full_pipeline = make_union(num_pipeline, cat_pipeline)

X_train = full_pipeline.fit_transform(train)

knn = KNeighborsClassifier()
sgd = SGDClassifier()
rfc = RandomForestClassifier()

#KNN param grid: {'n_neighbors': [10, 15, 30], 'weights': ('uniform', 'distance')}
#RFC param grid: {'n_estimators': [10, 20, 50, 100], 'max_depth': [2, 3], 
 #                'max_features': ['sqrt', 'log2']}
param_grid = {'loss': ('hinge', 'log'), 'penalty': ('l1', 'l2', 'elasticnet'), 
              'alpha': [0.0001, 0.001, 0.01, 0.1]}

clf = GridSearchCV(sgd, param_grid, cv=3,)
clf.fit(X_train, y_train)

print(clf.best_params_)
model = clf.best_estimator_

print_statistics(model, X_train, y_train)




