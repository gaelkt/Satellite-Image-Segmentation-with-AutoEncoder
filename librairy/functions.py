#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file contains some useful function that we call from the main file Train.py. You don't have to run this file by yourself
Requirements:
    1. PIL
    
    2. OSGEO

@author: Gael Kamdem De Teyou for IRD/Geoazur/Luxcarta
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal
import tensorflow as tf



# PLot 4 images simultaneously
def plot_sample(Image_1, Image_2, Image_3, Image_4, Label_1='', Label_2='', Label_3='', Label_4=''):
    size = 45
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0, 0].imshow(Image_1)
    ax[0, 0].set_title(Label_1, fontweight="bold", size=size)
    ax[0, 1].imshow(Image_2)
    ax[0, 1].set_title(Label_2, fontweight="bold", size=size)
    ax[1, 0].imshow(Image_3)
    ax[1, 0].set_title(Label_3, fontweight="bold", size=size)
    ax[1, 1].imshow(Image_4)
    ax[1, 1].set_title(Label_4, fontweight="bold", size=size)

    return

# PLot time series data
def plot_training(perf):
    iteration = np.arange(len(perf))
    plt.plot(iteration, perf,'g-')
    #plt.axis([0, 40, 0, 6])
    plt.show()
    return

def generator_(arguments):
    """
    Read a patch and the corresponding mask 
    1. Input
        - folder_patch: String. LOcation of patches
        - folder_mask: String. LOcation of masks
    
    2. Image_List: List of string. Names pf different images
    
    3. classes: int32. NUmber of classes
    
    4. Output: (float 32, float32) of size [patch_size, patch_size, nb_channel] and [patch_size, patch_size, classes].
            Processed patches and masks

    """
    folder_patch, folder_mask, Image_List, classes = arguments[0], arguments[1], arguments[2], arguments[3]
    while True:
        for name in Image_List:
            
            # We remove the byte identifier b' on the string name
            name = name.decode('utf-8')
            
            #Getting each filename name
            filename_mask = folder_mask + name
            filename_patch = folder_patch + name
            
            #Opening each file
            img_patch = np.array(Image.open(filename_patch))
            img_mask = np.array(Image.open(filename_mask))
            
            if classes == 1:
                # We add one extra dimension to have a shape [size_x, size_y, 1]
                img_mask = np.expand_dims(img_mask, axis=2)
                
            else:
                # The shape must be [size_x, size_y, classes]
                img_mask = ''
            # Normalization of patches and masks
            yield img_patch/255.0, img_mask/255.0

def cross_entropy_loss(mask_hard_true_resh, mask_soft_pred_resh, classes):
    """
    Compute the cross entropy between the predicted mask and the true mask
    1. mask_hard_true_resh: True mask hard coded and reshaped to a vector with shape [batch_size*patch_size*patch_size]. Each element contains the index number of the true class.
        - For e.g  with classes = 3: mask_hard_true_resh[0] = 1, mask_hard_true_resh[1] = 2, mask_hard_true_resh[2] = 0, mask_hard_true_resh[2] = 2
        - For e.g  with classes = 1: mask_hard_true_resh[0] = 1, mask_hard_true_resh[1] = 0, mask_hard_true_resh[2] = 0, mask_hard_true_resh[2] = 1
    
    2. reshaped_pred: Predicted mask soft coded with shape [batch_size*patch_size*patch_size, classes]. The elements are logits at the output. Not the probabilities
     . Each row must contains a vector representing the features maps or logits for each pixel.
     Softmax has not been applied earlier to have probability distribution.
         - For e.g with classes = 3: mask_soft_pred_resh[0] = [-0.8 23.2 45.6], mask_soft_pred_resh[1] = [47.5 -31.2 24.9] and etc...
         - For e.g with classes = 1: mask_soft_pred_resh[0] = 54.6, mask_soft_pred_resh[1] = -785.5 and etc...
         
    3. Don't call this function with the output of a softmax or a sigmoid. 
        mask_soft_pred_resh is expected to be unscaled logits since we are going to perform either a sigmoid or a softmax internally
        
    4. Return float32: cross entropy

    """
    if classes == 1:
        
        # We have a binary classification
        cost_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = mask_hard_true_resh, logits = mask_soft_pred_resh)
        
    else:
        # We have a multi class classification
        cost_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = mask_hard_true_resh, logits = mask_soft_pred_resh)
        
    return tf.reduce_mean(cost_entropy)
    
    


def soft_intersection_union(mask_hard_true_resh, mask_soft_pred_resh, classes):
    """    
    calculate intersection over union log loss
    
    1. mask_hard_true_resh: True mask hard coded and reshaped to a vector with shape [batch_size*patch_size*patch_size]
    
    2. mask_soft_pred_resh: Predicted mask soft coded at the output of UNET with shape [batch_size*patch_size*patch_size, classes]. The elements are logits
    Since elements of mask_soft_pred_resh are unscaled logits, we need to translate them into probabilities with Sigmoid or Softmax
    
   
    """
    if classes ==1:
        mask_soft_pred_resh = tf.nn.sigmoid(mask_soft_pred_resh)
    else:
        mask_soft_pred_resh = tf.nn.sigmoid(mask_soft_pred_resh)
        
    intersection = tf.multiply(mask_soft_pred_resh, mask_hard_true_resh)
    union = tf.subtract(tf.add(mask_soft_pred_resh, mask_hard_true_resh), intersection)
        
    intersection_total = tf.reduce_sum(intersection)
    union_total = tf.reduce_sum(union)
        
    intersection_total += 1e-16
    union_total += 1e-16
    
        
    return tf.multiply(tf.constant(-1.0), tf.log(tf.divide(intersection_total, union_total)))
    


def accuracy_metric(mask_hard_pred, mask_hard_true):
    """
    Compute the accuracy in percentage between the predicted mask and the true mask
    1. mask_hard_true: True mask with the shape of the patch [batch_size, patch_size, patch_size]. Each element contains the index number of the true class.
        - For e.g  with classes = 3: mask_hard_true[0, 25, 32] = 1, mask_hard_true[1, 98, 251] = 2, mask_hard_true[2, 254, 232] = 0
        - For e.g  with classes = 1: mask_hard_true[0, 25, 32] = 1, mask_hard_true[1, 98, 251] = 0, mask_hard_true[2, 254, 232] = 1
    
    2. mask_hard_pred: Predicted mask that contains the predicted classes for each pixel
     
         - For e.g with classes = 3: mask_hard_pred[0, 25, 32] = 1, mask_hard_pred[1, 98, 251] = 2, mask_hard_pred[2, 254, 232] = 0
         - For e.g with classes = 1: mask_hard_pred[0, 25, 32] = 1, mask_hard_pred[1, 98, 251] = 0, mask_hard_pred[2, 254, 232] = 0
         
    3. The function does element wise comparisons

    """
    acc = tf.cast(tf.equal(mask_hard_pred, mask_hard_true), 'float')
    return tf.reduce_mean(acc)
        
def iou_metric(mask_hard_true, mask_hard_pred):
    """    
  
    calculate intersection over union metric
   
    """

        
    intersection = tf.multiply(mask_hard_true, mask_hard_pred)
    union = tf.subtract(tf.add(mask_hard_true, mask_hard_pred), intersection)
        
    intersection_total = tf.reduce_sum(intersection)
    union_total = tf.reduce_sum(union)
        
    intersection_total += 1e-16
    union_total += 1e-16
    
        
    return tf.divide(intersection_total, union_total)
           
def process_patches(patch, mean, is_training):
        """
        1. normalize patches with the formula:
            x-mean
        
        2. apply atmostpheric correction
        
        """
        
        return patch/255


            

