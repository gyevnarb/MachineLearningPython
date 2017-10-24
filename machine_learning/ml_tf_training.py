# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:19:42 2017

@author: Bálint
"""

import numpy as np
import tensorflow as tf

tf.reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.float32, shape=(None, n_outputs), name='y')

is_training = tf.placeholder(tf.float32, shape=(), name='is_training')
bn_params = {'is_training': is_training, 'decay': 0.99, 'updates_collections': None}

from tensorflow.contrib.layers import fully_connected

with tf.name_scope('dnn'):
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            normalizer_fn=batch_norm,
            normalizer_params=bn_params):
        hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
        logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
    
with tf.name_scope('loss')   :
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    
learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./tmp/data/')

n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for X_batch, y_batch in zip(X_batches, y_batches):
            sess.run(training_op,
                     feed_dict={is_training: True, X: X_batch, y: y_batch})
        accuracy_score = accuracy.eval(
                feed_dict={is_training: False, X: X_test_scaled, y: y_test}))
        print(accuracy_score)