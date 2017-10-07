# Copyright 2017 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Implementation of the Multilayer Perceptron using TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import sys
import tensorflow as tf


class MLP:
    """Implementation of the Multilayer Perceptron using TensorFlow"""

    def __init__(self, alpha, batch_size, node_size, num_classes, num_features):
        """Initialize the MLP model

        Parameter
        ---------
        alpha : float
          The learning rate to be used by the neural network.
        batch_size : int
          The number of batches to use for training/validation/testing.
        node_size : int
          The number of neurons in the neural network.
        num_classes : int
          The number of classes in a dataset.
        num_features : int
          The number of features in a dataset.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.node_size = node_size
        self.num_classes = num_classes
        self.num_features = num_features

        def __graph__():
            """Build the inference graph"""

            with tf.name_scope('input'):
                # [BATCH_SIZE, NUM_FEATURES]
                x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.num_features], name='x_input')

                # [BATCH_SIZE]
                y_input = tf.placeholder(dtype=tf.uint8, shape=[None], name='y_input')

                # [BATCH_SIZE, NUM_CLASSES]
                y_onehot = tf.one_hot(indices=y_input, depth=self.num_classes, on_value=1, off_value=0, name='y_onehot')

            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            first_hidden_layer = {'weights': self.weight_variable([self.num_features, self.node_size]),
                                  'biases': self.bias_variable([self.node_size])}

            second_hidden_layer = {'weights': self.weight_variable([self.node_size, self.node_size]),
                                  'biases': self.bias_variable([self.node_size])}

            third_hidden_layer = {'weights': self.weight_variable([self.node_size, self.node_size]),
                                  'biases': self.bias_variable([self.node_size])}

            output_layer = {'weights': self.weight_variable([self.node_size, self.num_classes]),
                            'biases': self.bias_variable([self.num_classes])}

            first_layer = tf.add(tf.matmul(x_input, first_hidden_layer['weights']), first_hidden_layer['biases'])

            first_layer = tf.nn.relu(first_layer)

            second_layer = tf.add(tf.matmul(first_layer, second_hidden_layer['weights']), second_hidden_layer['biases'])

            second_layer = tf.nn.relu(second_layer)

            with tf.name_scope('training_ops'):
                with tf.name_scope('weights'):
                    weight = tf.get_variable(name='weights',
                                             initializer=tf.random_normal([self.num_features, self.num_classes],
                                                                          stddev=0.01))
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.get_variable(name='biases', initializer=tf.constant([0.1], shape=[self.num_classes]))
                    self.variable_summaries(bias)
                with tf.name_scope('Wx_plus_b'):
                    pass

            self.x_input = x_input
            self.y_input = y_input
            self.y_onehot = y_onehot
            self.learning_rate = learning_rate

        sys.stdout.write('\n<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    @staticmethod
    def weight_variable(shape):
        """Initialize weight variable

        Parameter
        ---------
        shape : list
          The shape of the initialized value.

        Returns
        -------
        The created `tf.get_variable` for weights.
        """
        initial_value = tf.random_normal(shape=shape, stddev=0.01)
        return tf.get_variable(name='weights', initializer=initial_value)

    @staticmethod
    def bias_variable(shape):
        """Initialize bias variable

        Parameter
        ---------
        shape : list
          The shape of the initialized value.

        Returns
        -------
        The created `tf.get_variable` for biases.
        """
        initial_value = tf.constant([0.1], shape=shape)
        return tf.get_variable(name='biases', initializer=initial_value)

    @staticmethod
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)