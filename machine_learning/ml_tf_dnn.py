# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 14:36:28 2017

@author: Bálint
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, W) + b
        if activation == 'relu':
            return tf.nnrelu(z)
        else:
            return z

def leaky_relu(z, name=None):
    return tf.maximum(0.01*z, z, name=name)

from tensorflow.contrib.layers import fully_connected

with tf.name_scope('dnn'):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1", activation_fn=leaky_relu)
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2", activation_fn=tf.nn.elu)
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
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, 'Training accuracy:', acc_train, 'Test accuracy:', acc_test)
    
    save_path = saver.save(sess, './my_model_final.ckpt')
            
#X_train = mnist.train.images
#X_test = mnist.test.images
#y_train = mnist.train.labels.astype("int")
#y_test = mnist.test.labels.astype("int")
#
#config = tf.contrib.learn.RunConfig(tf_random_seed=42) # not shown in the config
#
#feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
#dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10,
#                                         feature_columns=feature_cols, config=config)
#dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
#dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)
#
#
#y_pred = dnn_clf.predict(X_test)
#print(accuracy_score(y_test, y_pred['classes']))

