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

import tensorflow as tf


class MLP:
    """Implementation of the Multilayer Perceptron using TensorFlow"""

    def __init__(self, alpha, batch_size, num_classes, num_features):
        """Initialize the MLP model

        Parameter
        ---------
        alpha : float
        batch_size : int
        num_classes : int
        num_features : int
        """
        self.alpha = alpha
        self.batch_size = batch_size
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
                y_onehot = tf.placeholder(indices=y_input, depth=self.num_classes, on_value=1, off_value=0,
                                          name='y_onehot')

            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

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