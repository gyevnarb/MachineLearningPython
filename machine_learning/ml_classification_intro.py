# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:26:59 2017

@author: Bálint
"""
import numpy as np
import numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure()
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()

def plot_digit(digit_intensities):
    some_digit_image = digit_intensities.reshape(28, 28)
    plt.figure()
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, 
           interpolation='nearest')
    plt.axis('off')
    plt.show()


mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']

some_index = rnd.randint(0,10000)
some_digit = X[some_index]
plot_digit(some_digit)

X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

param_grid = [
        {'weights': ['uniform', 'distance']}, {'n_neighbors': [5, 8, 12]}
    ]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, scoring='neg_log_loss', 
                           n_jobs=4, cv=3, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(cross_val_score(best_model, X_train, y_train, cv=3, scoring='accuracy'))

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy'))

never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy'))

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5, y_train_pred))
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
print(f1_score(y_train_5, y_train_pred))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)


#sgd_clf.fit(X_train, y_train)
#print(sgd_clf.decision_function([some_digit]))
#
#forest_clf = RandomForestClassifier()
#scaler = StandardScaler()
#X_train_scaled  = scaler.fit_transform(X_train.astype(np.float64))
#forest_clf.fit(X_train_scaled, y_train)
#print(forest_clf.predict_proba([some_digit]))
#print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
#
#knn_clf = KNeighborsClassifier()
#
#noise = rnd.randint(0, 100, (len(X_train), 784))
#X_train_mod = X_train + noise
#noise = rnd.randint(0, 100, (len(X_test), 784))
#X_test_mod = X_test + noise
#y_train_mod = X_train
#y_test_mod = X_test
#plot_digit(X_test_mod[some_index])
#
#knn_clf.fit(X_train_mod, y_train_mod)
#clean_digit = knn_clf.predict(X_test_mod[some_index])
#plot_digit(clean_digit)







