import numpy as np
import tensorflow as tf



class TextCNN(object):
    def __init__(self, num_classes, num_layers):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, 256, 256], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        x_image = tf.reshape(self.input_x, [-1, 256, 256, 1])
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters = 32
        num_filters_prev = 1
        pooled_prev = x_image
        filter_size = 3
        for i in range(num_layers):
            with tf.name_scope('conv-maxpool-%d' % i):
                # Convolution Layer
                filter_shape = [filter_size, filter_size, num_filters_prev, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(pooled_prev, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name='pool')
                pooled_outputs.append(pooled)
                num_filters_prev = num_filters
                num_filters *= 2
                pooled_prev = pooled

        # Combine all the pooled features
        pooled_size_total = 256 >> num_layers
        num_filters_total = pooled_size_total*pooled_size_total*num_filters_prev
        #self.h_pool = tf.concat(pooled_prev, 3)
        self.h_pool_flat = tf.reshape(pooled_prev, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable('W', shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,
                                                             logits=self.scores)  # only named arguments accepted
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
