# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:02:46 2017

@author: Bálint
"""
import numpy.random as rnd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from scipy.stats import mode

def print_statistics(model, X_train, y_train):
    some_index = rnd.randint(0, len(X_train))
    some_person = X_train[some_index].reshape(1, -1)
    print('Prediction: %s True: [%s]' % (model.predict(some_person), y_train[some_index]))
    y_train_predict = cross_val_predict(model, X_train, y_train, cv=3)
    print('Accuracy:', cross_val_score(model, X_train, y_train, cv=3))
    print('Precision:', precision_score(y_train, y_train_predict))
    print('Recall:', recall_score(y_train, y_train_predict))
    print('F1 Score:', f1_score(y_train, y_train_predict))


moons, labels = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(moons, labels)

param_grid = {'max_leaf_nodes': [2, 3, 4, 5], 'max_depth': [2, 3, 4]}

dtc = DecisionTreeClassifier()
clf = GridSearchCV(dtc, param_grid, cv=3)
clf.fit(X_train, y_train)

print(clf.best_params_)
model = clf.best_estimator_
some_ex = X_train[20].reshape(1,-1)
print_statistics(model, X_test, y_test)

rs = ShuffleSplit(n_splits=1000, train_size=100, test_size=0.0, random_state=42)
dtc_clf = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=4)
y_forest_pred = []
for train_index, test_index in rs.split(X_train, y_train):
    X_subset = X_train[train_index]
    y_subset = y_train[train_index]
    dtc_clf.fit(X_subset, y_subset)
    y_forest_pred.append(mode(dtc_clf.predict(X_test)))





