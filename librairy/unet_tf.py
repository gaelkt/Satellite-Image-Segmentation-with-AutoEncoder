# -*- coding: utf-8 -*-
"""
This file defines the UNET architecture and some functions Tf needs

@author: Gael Kamdem De Teyou
"""

import numpy as np
import tensorflow as tf


# Weights initiqlization
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

# Biais Initialization
def create_biases(size):
    return tf.Variable(tf.constant(0.1, shape=[size]))

# Dropout may be used
def dropout(input, keep_prob):
        return tf.nn.dropout(input, keep_prob)

# Single Convolution layer
def conv_layer(input, num_input_channels, conv_filter_size, num_filters, name_oper, padding='SAME', activation='relu', training=True):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding, name = name_oper)
    layer += biases
    
    if training:
            layer = tf.layers.batch_normalization(layer, axis = -1, momentum = 0.9, training = training)

    if activation == 'relu':
        layer = tf.nn.relu(layer)
    elif activation == 'sigmoid':
        layer = tf.nn.sigmoid(layer)
    elif activation == 'softmax':
        layer = tf.nn.softmax(layer)
    
    return layer


# Pooling layer. The size of input is [batch_size, size_x, size_x, nb_channel] 
def pool_layer(input, name_oper, padding='SAME'):
    return tf.nn.max_pool(value=input,
                          ksize = [1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding=padding,
                          name = name_oper)


# Transposed convolution
def un_conv(input, num_input_channels, conv_filter_size, num_filters, feature_map_size, batch_size, name_oper, padding='SAME',relu=True, training=True):


    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
    biases = create_biases(num_filters)
    
    if training:
        batch_size_0 = batch_size
    else:
        batch_size_0 = 1
    
    #output_shape = tf.stack([batch_size, feature_map_size, feature_map_size, num_filters])
    layer = tf.nn.conv2d_transpose(value=input, filter=weights,
                                   output_shape=tf.stack([tf.shape(input)[0], feature_map_size, feature_map_size, num_filters]),
                                   strides=[1, 2, 2, 1],
                                   padding=padding,
                                   name = name_oper)
    layer += biases
    
#    if training:
#        layer = tf.layers.batch_normalization(layer, fused = True, axis = [1, 2], training = training)
        
    if relu:
        layer = tf.nn.relu(layer)
        
    return layer


# The UNET
def create_unet(input, n_channel, kernel_size_down, kernel_size_up, n_filters, classes, training=True):

    
    img_size = tf.shape(input)[1]
    
    batch_size = np.shape(input)[0]
    

###### Contracting Path
    conv1 = conv_layer(input, n_channel, kernel_size_down, n_filters, 'conv1', training=True)
    conv2 = conv_layer(conv1, n_filters, kernel_size_down, n_filters, 'conv2', training=True)
    pool2 = pool_layer(conv2,  'pool2')
    #pool2 = dropout(pool2, keep_prob)    # Dropout included but may be canceled
    
    conv3 = conv_layer(pool2, n_filters, kernel_size_down, n_filters*2, 'conv3', training=True)
    conv4 = conv_layer(conv3, n_filters*2, kernel_size_down, n_filters*2, 'conv4', training=True)
    pool4 = pool_layer(conv4,  'pool4')
    #pool4 = dropout(pool4, keep_prob) 
    
    conv5 = conv_layer(pool4, n_filters*2, kernel_size_down, n_filters*4, 'conv5', training=True)
    conv6 = conv_layer(conv5, n_filters*4, kernel_size_down, n_filters*4, 'conv6', training=True)
    pool6 = pool_layer(conv6,  'pool6')
    #pool6 = dropout(pool6, keep_prob) 

    conv7 = conv_layer(pool6, n_filters*4, kernel_size_down, n_filters*8, 'conv7', training=True)
    conv8 = conv_layer(conv7, n_filters*8, kernel_size_down, n_filters*8, 'conv8', training=True)
    pool8 = pool_layer(conv8,  'pool8')
    #pool8 = dropout(pool8, keep_prob)Created on Wed Dec 18 14:49:40 2019

    conv9 = conv_layer(pool8, n_filters*8, kernel_size_down, n_filters*16,  'conv9', training=True)
    conv10 = conv_layer(conv9, n_filters*16, kernel_size_down, n_filters*16, 'conv10', training=True)
    
###### Expanding Path

    conv11 = un_conv(conv10, n_filters*16, kernel_size_up, n_filters*8, img_size // 8, batch_size, 'conv11', training=True)
    merge11 = tf.concat(values=[conv8, conv11], axis = -1, name = 'merge11')

    conv12 = conv_layer(merge11, n_filters*16, kernel_size_down, n_filters*8, 'conv12', training=True)
    conv13 = conv_layer(conv12, n_filters*8, kernel_size_down, n_filters*8, 'conv13', training=True)

    conv14 = un_conv(conv13, n_filters*8, kernel_size_up, n_filters*4, img_size // 4, batch_size, 'conv14', training=True)
    merge14 = tf.concat([conv6, conv14], axis=-1, name = 'merge14')

    conv15 = conv_layer(merge14, n_filters*8, kernel_size_down, n_filters*4, 'conv15', training=True)
    conv16 = conv_layer(conv15, n_filters*4, kernel_size_down, n_filters*4, 'conv16', training=True)

    conv17 = un_conv(conv16, n_filters*4, kernel_size_up, n_filters*2, img_size // 2, batch_size, 'conv17', training=True)
    merge17 = tf.concat([conv17, conv4], axis=-1, name = 'conv17')

    conv18 = conv_layer(merge17, n_filters*4, kernel_size_down, n_filters*2, 'conv18', training=True)
    conv19 = conv_layer(conv18, n_filters*2, kernel_size_down, n_filters*2, 'conv19', training=True)

    conv20 = un_conv(conv19, n_filters*2, kernel_size_up, n_filters*1, img_size, batch_size, 'conv20', training=True)
    merge20 = tf.concat([conv20, conv2], axis=-1, name = 'merge20')

    conv21 = conv_layer(merge20, n_filters*2, kernel_size_down, n_filters*1, 'conv21', training=True)
    conv22 = conv_layer(conv21, n_filters*1, kernel_size_down, n_filters*1, 'conv22', training=True)
    
    
    # Output shape is [batch_size, size_x, size_y, classes]
    # No activation for output
    if classes ==1:
        # The sigmoid will be applied during loss calculation with tf.nn.sigmoid_cross_entropy_with_logits
        output = conv_layer(conv22, n_filters*1, 1, classes, 'conv23', 'SAME', 'nothing', training = False)
    else:
        # The sparse softmax will be applied during loss calculation 
        output = conv_layer(conv22, n_filters*1, 1, classes, 'conv23', 'SAME', 'nothing', training = False)

    return output