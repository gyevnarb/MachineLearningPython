# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:55:35 2017

@author: Bálint
"""

#General imports
import tensorflow as tf
import numpy as np
import os

#Reset default graph to avoid variable collision
tf.reset_default_graph()

#Setup data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./tmp/data/')

val_idx = np.in1d(mnist.validation.labels, range(1,5))
test_idx = np.in1d(mnist.test.labels, range(1,5))

X_val = mnist.validation.images[val_idx]
y_val = mnist.validation.labels[val_idx]
X_test = mnist.test.images[test_idx]
y_test = mnist.test.labels[test_idx]

#Saver params
save_root = './saves'

#Logging params
from datetime import datetime
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = './tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

#Construction related hyperparams
n_inputs = 28 * 28
n_layers = 5
n_hidden = 100
n_outputs = 5

#Execution related hyperparams
batch_size = 50
n_epochs = 40
n_batches = mnist.train.num_examples // batch_size
prev_val_score = 0
since_last_best_step = 0
stop = False

#Traing hyperparams
bn_momentum = 0.99
learning_rate = 0.01
step_stop = 5000
dropout_rate = 0.5

#Graph placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')

from functools import partial

with tf.name_scope('dnn'):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    
    cr_dense     = partial(tf.layers.dense, units=n_hidden, 
                               kernel_initializer=he_init, activation=tf.nn.elu)
    
    cr_batchnorm = partial(tf.layers.batch_normalization, training=training, 
                               momentum=bn_momentum)
    
    cr_dropout   = partial(tf.layers.dropout, rate=dropout_rate, 
                               training=training)
    
    hidden1 = cr_dropout(cr_batchnorm(cr_dense(X, name='hidden_1')))
    hidden2 = cr_dropout(cr_batchnorm(cr_dense(X, name='hidden_2')))
    hidden3 = cr_dropout(cr_batchnorm(cr_dense(X, name='hidden_3')))
    hidden4 = cr_dropout(cr_batchnorm(cr_dense(X, name='hidden_4')))
    hidden5 = cr_dropout(cr_batchnorm(cr_dense(X, name='hidden_5')))
    logits  = cr_dropout(cr_batchnorm(cr_dense(hidden5, units=n_outputs,
                                               name='outputs', activation=None)))
            
with tf.name_scope('loss'):
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xe, name='loss')
tf.summary.scalar('cross_entropy', loss)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('accuracy'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs): 
        if os.path.isfile(save_root+'/my_model_final.ckpt'):
            print('Restoring current model:')
            #saver.restore(sess, save_root+'/my_model_final.ckpt')
        if epoch % 10 == 0:
            saver.save(sess, save_root+'/my_model_chckpoint.ckpt')
                    
        test_accuracy_score = accuracy.eval(
                feed_dict={training: False, X: X_test, y: y_test})
        print('Epoch:', epoch+1, 'Accuracy score:', test_accuracy_score)
        
        for iteration in range(n_batches):
            step = epoch * n_batches + iteration
            since_last_best_step += 1
            
            #Train the neural net
            X_batch, y_batch = mnist.train.next_batch(batch_size)   
            batch_idx = np.in1d(y_batch, range(1,5))
            X_batch = X_batch[batch_idx]
            y_batch = y_batch[batch_idx]
            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, y: y_batch})
            
            if since_last_best_step > step_stop:
                print('Early stopping')
                stop = True
                break
            
            if step % 50 == 0:
                val_accuracy_score = accuracy.eval(
                    feed_dict={training: False, X: X_val, y: y_val})
                print('\tStep:', step, 
                      'Current validation score:', val_accuracy_score,
                      'Previous validation score', prev_val_score,
                      'Steps since best', since_last_best_step)
                if val_accuracy_score > prev_val_score:
                    since_last_best_step = 0
                    prev_val_score = val_accuracy_score
                    saver.save(sess, save_root+'/step_model.ckpt')
            
            if step % 100 == 0:
                test_summary  = sess.run(merged_summary, feed_dict=
                                         {training: False, X: X_test, y: y_test})
                file_writer.add_summary(test_summary, epoch)
        if stop: 
            break
      
    print('Saving model')
    saver.save(sess, save_root+'/my_model_final.ckpt')
    
print('Model training complete')
file_writer.close()