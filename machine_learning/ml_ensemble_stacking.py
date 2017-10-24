# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:27:21 2017

@author: Bálint
"""

import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("MNIST")
data, target = mnist['data'], mnist['target']
print(data)
X, y = data[:40000], target[:40000]
X_valid, y_valid = data[40000:50000], target[40000:50000]
X_test, y_test = data[50000:60000], target[50000:60000]