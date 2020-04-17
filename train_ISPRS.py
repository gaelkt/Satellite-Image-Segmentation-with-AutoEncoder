#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is the main file used to train the UNET model.
Requirements:
    1. First, you need to use the file generate_dataset to generate pacthes from high resolution images
    
    2. folder_features_sampled is the folder where training patches are stores. Same for folder_labels_sampled, folder_features_validation and folder_labels_validation

    3. You need the folder /libs that contains the TensorFlow model unet_tf.py and some utilities functions functions.py 
    
    4. You need the folder ./Save/Sampled/ where weights can be saved and restored. For transfer learning you will find initial weights in this folder
    
    5. You need NVIDIA GPU
    
    6. DL Dependencies: Tensorflow-gpu, CUDA 10.0 and Cudnn 7.0
    
    7. Other dependencies: GDAL
@author: Gael Kamdem De Teyou for IRD/Geoazur/Luxcarta
"""

#--------------------------------  Packages needed   -------------------------------------------
#-----------------------------------------------------------------------------------------------
import gc

# We delete all variable
gc.collect()


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline
import tensorflow as tf
import scipy.io
import random
import multiprocessing
import sys
sys.path.insert(0, './libs/')
from unet_tf import create_unet
from functions import plot_sample, cross_entropy_loss, soft_intersection_union, accuracy_metric, iou_metric, plot_training

import os
import time
from sklearn.metrics import accuracy_score
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk, square
from skimage.filters import gaussian
from skimage import data, exposure, img_as_float

from scipy import ndimage

print('------------------------------------------------------------------------------------------')
print('---------------------------  Packet Imported Successfully --------------------------------')
print('------------------------------------------------------------------------------------------')



"""
We make sure GPUs are available:
    1. gpus: list containing  the number of available GPU
    
    2. We configure GPU to enable memory growth when simulations are runnings

"""

# We make sure GPUs are available and memory growth is activated
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

"""
#--------------------------------  Environment variables -----------soft_intersection_union-------------------------------
#--------------------------------------------------------------------------------------------------
# Specify GPUs that you want to be visible, not device order
# We use GPU 0 and 1. You can change this to 0, 1, ..., len(gpus) etc...
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="1"


# We reset the graph
#tf.reset_default_graph()
tf.keras.backend.clear_session()  # For tensorflow 2.0

"""
#--------------------------------------------------------------------------------------------------
#--------------------------------  Parameters of the simulation -----------------------------------
#--------------------------------------------------------------------------------------------------
"""

# Location of patches and mask
# Training set
#folder_features_sampled, folder_labels_sampled = "../../../Documents/GeoAzur/dataset/Luxcarta/train/augmented/patch/", "../../../Documents/GeoAzur/dataset/Luxcarta/train/augmented/mask/"
#folder_patch_train, folder_mask_train = "../../../Documents/GeoAzur/dataset/Luxcarta/train/augmented_II/patch/", "../../../Documents/GeoAzur/dataset/Luxcarta/train/augmented_II/mask/"
#folder_patch_train, folder_mask_train = "../../../Documents/GeoAzur/dataset/Luxcarta/train/patch/", "../../../Documents/GeoAzur/dataset/Luxcarta/train/mask/"
folder_patch_train, folder_mask_train = "../../../dataset_ISPRS/sampled/augmented/patch/", "../../../dataset_ISPRS/sampled/augmented/mask/"

# Validation set
folder_patch_val, folder_mask_val = "../../../dataset_ISPRS/sampled/valid/patch/", "../../../dataset_ISPRS/sampled/valid/mask/"


# Parameters of the model

# Size of each patch
# Each TIF image is divided into patches of size [patch_size, patch_size, nb_channel] with a stride 
patch_size, nb_channel, stride, classes = 256, 3, 256, 1   # size of new sampled image and stride for patches


# Unet Parameters
n_filters, kernel_size_down, kernel_size_up = 32, 3, 2

# Training Parameters
batch_size, epoch = 40, 20

# Number of training and validation data
Nb_samples_train = len(os.listdir(folder_patch_train))
step_per_epok = int(Nb_samples_train // batch_size)

Nb_samples_val = len(os.listdir(folder_mask_val))
step_per_epoch_val = int(Nb_samples_val // batch_size)


# Nb of iterations
N_iterations = epoch*step_per_epok

# Arrays to store the performances during training and validation
accuracy_iter, loss_iter, IoU_iter = np.zeros((N_iterations)), np.zeros((N_iterations)), np.zeros((N_iterations))

accuracy_val, accuracy_val_2, loss_val, IoU_val = np.zeros((step_per_epoch_val)), np.zeros((step_per_epoch_val)), np.zeros((step_per_epoch_val)), np.zeros((step_per_epoch_val))

# Threshold for decision
threshold = 0.04

# Decimals for printing results
decimals = 3

gamma = 0.5

sigma = 1.35

selem = square(2)
o, m, n = 5, 1, 1
Total = 4*n + 4*m + o
R = 1.0/Total

filt = R*np.array([[n, m, n], [m, o, m], [n, m, n]])

filt = 0.25*np.ones((2, 2))


print('------------------------------------------------------------------------------------------')
print('----------------------------------  Training info ----------------------------------------')
print('------------------------------------------------------------------------------------------')
print('---------------  Nb of train patches = ' + str(Nb_samples_train) + ' and Nb of validation patches = ' + str(Nb_samples_val) + '   --------------')
print('---------------  Nb of iterations =' + str(N_iterations) + '  Epochs = '+ str(epoch) + '  Steps per Epochs = ' + str(step_per_epok) + ' ----------------------')
print('------------------------------------------------------------------------------------------')

"""
#--------------------------------------------------------------------------------------------------
#--------------------------------   Counting Patches and Masks  -----------------------------------
#--------------------------------------------------------------------------------------------------
"""

# We check patches and masks and we shuffle them
# Image_Name_List is the list containing the name of all patches

Image_Name_List_Patch, Label_Name_List_Mask = os.listdir(folder_patch_train), os.listdir(folder_mask_train)  # Name of all patches and masks for training
Image_Name_List_val, Label_Name_LIst_val = os.listdir(folder_patch_val), os.listdir(folder_mask_val)  # Name of all patches and masks for validation

Label_Name_List_Mask_ = [name.replace('mask', 'patch') for name in Label_Name_List_Mask]
Label_Name_List_Mask_.sort()
Image_Name_List_Patch.sort()
if Image_Name_List_Patch != Label_Name_List_Mask_:
    print('------------------------------------------------------------------------------------------')
    sys.exit(" -----------------    There are mismatches between patches and masks ----------------------")
else:
    print('------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------')
    print('---------------------------    Patches and Masks are OK ----------------------------------')
    print('------------------------------------------------------------------------------------------')
    print('------------------Total number of training Patches is  ', len(Image_Name_List_Patch))
    print('---------------------------    Patches are being randomized    ---------------------------')
    random.shuffle(Image_Name_List_Patch) 
    random.shuffle(Image_Name_List_Patch) 
    print('---------------------------  Randomization Success   -------------------------------------')
    print('------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------')


print('------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------')
print('---------------------------      Generating Batches  -------------------------------------')
print('------------------------------------------------------------------------------------------')

"""
#--------------------------------------------------------------------------------------------------
#---------------------------------------  Data Pipeline   -----------------------------------------
#--------------------------------------------------------------------------------------------------

We implement a tf data pipeline that define how data will be read, fed to the deep neural network and how this tasks
will be distributed accross different threads 

    1.  generator() is an iterator that produces one single patch and the corresponding mask. The patch is processed
    with a custom function preprocessing() to normalize the data before being returned
    
    2. Here will also tell the data pipeline the batch_size samples and to prefecth some of them to avoid latency during processing
    
    3. The data are shuffled during each mini batch
    
    4. For this purpose, we use tf.Data API
    
    For more info read this:
        https://medium.com/@nimatajbakhsh/building-multi-threaded-custom-data-pipelines-for-tensorflow-f76e9b1a32f5
    
"""

def generator_val(Image_Name):
    while True:
        for name in Image_Name:
            
            name = name.decode('utf-8')
            
            #Getting each filename name
            filename_patch = folder_patch_val + name
            filename_mask = folder_mask_val + name.replace('patch', 'mask')
            
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
            img_patch = exposure.adjust_gamma(img_patch, gamma)
            
            #img_mask = gaussian(img_mask, sigma=1.5)
            #img_mask = dilation(img_mask)
            #img_mask = gaussian(img_mask, sigma=0.1, truncate=1/2)
            
            img_mask[:, :,0] = ndimage.grey_dilation(img_mask[:, :,0], footprint=filt)
            
            
            yield img_patch/255.0, np.round(img_mask/255)

def generator_train(Image_Name):
    while True:
        for name in Image_Name:
            
            name = name.decode('utf-8')
            
            #Getting each filename name
            filename_patch = folder_patch_train + name
            filename_mask = folder_mask_train + name.replace('patch', 'mask')
            
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
            # Gamma correction of the image
            img_patch = exposure.adjust_gamma(img_patch, gamma)
            
            # Dilation of roads line
            #img_mask = gaussian(img_mask, sigma=1.5)
            #img_mask = dilation(img_mask)
            
            img_mask[:, :,0] = ndimage.grey_dilation(img_mask[:, :,0], footprint=filt)
            
            #img_mask = gaussian(img_mask, sigma=0.1, truncate=1/2)
            

            yield img_patch/255.0, np.round(img_mask/255)


# Set up Multi-threaded Batch generator
            
# Parameters for multi-threading
output_types = (tf.float32, tf.float32)  # Types of generetor outputs 
CPU = multiprocessing.cpu_count()        # NUmber of available CPU on the computer
Thread = 2*CPU-5                                  # There is 2 Threads/CPU and we leave 5 threads unused to avoid overflow
Thread = 10

# We generate Thread different generators and we interleave them

# For training
generator_size = len(Image_Name_List_Patch) // Thread
arguments = [[Image_Name_List_Patch[m*Thread + r] for m in range(generator_size)] for r in range(Thread)]
Dataset = tf.data.Dataset

# Converting our generatf.datator to a data pipeline and using multi-Threading 
dataset_train = Dataset.from_tensor_slices(arguments)
dataset_train = dataset_train.interleave(lambda x: Dataset.from_generator(generator_train, output_types=output_types, args=(x,)),
                   cycle_length=Thread,
                   block_length=1,
                   num_parallel_calls=Thread)

# We shuffle data, generate mini batch of size batch_size  and prefetch them for optimization
dataset_train = dataset_train.shuffle(buffer_size=batch_size)
dataset_train = dataset_train.batch(batch_size = batch_size).prefetch(tf.data.experimental.AUTOTUNE)
iterator_train = dataset_train.make_one_shot_iterator()


# For validation
generator_size_val = len(Image_Name_List_val) // Thread
arguments_val = [[Image_Name_List_val[m*Thread + r] for m in range(generator_size_val)] for r in range(Thread)]
Dataset = tf.data.Dataset

# Converting our generatf.datator to a data pipeline and using multi-Threading 
dataset_val = Dataset.from_tensor_slices(arguments_val)
dataset_val = dataset_val.interleave(lambda x: Dataset.from_generator(generator_val, output_types=output_types, args=(x,)),
                   cycle_length=Thread,
                   block_length=1,
                   num_parallel_calls=Thread)

# We shuffle data, generate mini batch of size batch_size  and prefetch them for optimization
dataset_val = dataset_val.shuffle(buffer_size=batch_size)
dataset_val = dataset_val.batch(batch_size = batch_size).prefetch(tf.data.experimental.AUTOTUNE)
iterator_val = dataset_val.make_one_shot_iterator()


print('---------------------------         Generator Ok     -------------------------------------')

# Tensor for each mini batch
img_patch_train, img_mask_train = iterator_train.get_next()
img_patch_val, img_mask_val = iterator_val.get_next() 

print('---------------------------         Next Gen  Ok     -------------------------------------')


print('------------------------------------------------------------------------------------------')
print('-----------------------------  starting with TensorFlow ----------------------------------')
print('------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------')

#--------------------------------------------------------------------------------------------------
#----------------------------------  Loading UNET Model   -----------------------------------------
#--------------------------------------------------------------------------------------------------

# Placeholder variable: Patch and Label

"""
Inputs and Labels for the model:
    1. patch: Input patch with shape [batch_size, patch_size, patch_size, nb_channel]
    
    2. mask: Input Mask with shape [batch_size, patch_size, patch_size, classes]

"""
 
patch =tf.placeholder(tf.float32, shape=[None, None, None, nb_channel])
mask = tf.placeholder(tf.float32, shape=[None, None, None, classes])


"""
# output of the model with logits of shape = [batch_size, patch_size, patch_size, classes]

"""
#output = create_unet(patch, nb_channel, kernel_size_down, kernel_size_up, n_filters, classes, batch_size, training = True)
output = create_unet(patch, nb_channel, kernel_size_down, kernel_size_up, n_filters, classes, training = True)

#output = closing(output, selem)
    
if classes > 1:
    """ We have multi class segmentation:
        1. mask_hard_true: True mask hard coded with shape [batch_size, patch_size, patch_size]
           mask is by default soft coded with shape [batch_size, patch_size, patch_size, classes]

        2. mask_hard_true_resh: True mask hard coded and reshaped to a vector with shape [batch_size*patch_size*patch_size]
        folder_patch_val
        3. mask_hard_pred: Prediected mask hard coded with shape [batch_size, patch_size, patch_size]
            Equivalent to mask_hard_true
        
        4. mask_soft_pred_resh: Predicted mask soft coded with shape [batch_size*patch_size*patch_size, classes]. The elements are logits
        
    """

    # True masks
    mask_hard_true = tf.argmax(mask, axis = 3, name="mask_hard_true")      # [batch_size, patch_size, patch_size]. Each elt (pixel) of the matrice is the class of that pixels: 0 or 1, .., or classes-1
    mask_hard_true_resh= tf.reshape(mask_hard_true, [-1])                  # [batch_size*patch_size*patch_size]. Each elt (pixel) of the vector is the class of that pixels: 0, 1, .., or classes-1
    
    # Predictions
    mask_hard_pred = tf.argmax(output, axis = 3, name="mask_pred_class")   # [batch_size, patch_size, patch_size] Each elt of the matrice (pixel) corresponds the predicted class 0, 1, .., or classes-1
    mask_soft_pred_resh = tf.reshape(output, [-1, classes])                # [batch_size*patch_size*patch_size, classes] Each row (pixel) is the predicted logits for the pixel
    
    
else:
    """ classes = 1 and We have binary segmentation
        1. mask_hard_true: True mask hard coded with shape [batch_size, patch_size, patch_size]. The same as mask
    
    """

    # True labels
    mask_hard_true = tf.cast(tf.greater(mask,threshold), 'float')                                               # [batch_size, patch_size, patch_size]. Each elt (pixel) of the matrice is 0 or 1, the class of the pixel
    mask_hard_true_resh= tf.reshape(mask, [-1])                         # [batch_size*patch_size*patch_size].  Each row is 0 or 1 for the pixel
    
    # Predictions
    mask_hard_pred = tf.cast(tf.greater(output,threshold), 'float')    # [batch_size, patch_size, patch_size] Each elt of the matrice (pixel) corresponds the predicted class 0 or 1
    mask_soft_pred_resh= tf.reshape(output, [-1])                      #  [batch_size*patch_size*patch_size] Each elt of the vector (pixel) corresponds the calculated logit



# Loss
entropy_loss = cross_entropy_loss(mask_hard_true_resh, mask_soft_pred_resh, classes)
iou_loss = soft_intersection_union(mask_hard_true_resh, mask_soft_pred_resh, classes)
alpha = 0.6
loss = alpha*entropy_loss + (1-alpha)*iou_loss
print('--------------------------------------------------------------------------- ** Saved variables to Disk *****************************')
# Metrics
accuracy = accuracy_metric(mask_hard_true, mask_hard_pred)
IoU = iou_metric(mask_hard_true, mask_hard_pred)


# Static Optimizer 
#optimizer = tf.train.AdamOptimizer(le------    arning_rate=0.01).minimize(entropy)
#optimizer = tf.train.RMSPropOptimizer(0.001, decay = 0.95, momentum = 0.9, epsilon = 1e-10).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss)

# Optimizer with an exponential decay learning rate
lr = 0.1  
decay_steps = epoch*step_per_epok
decay = 0.95
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay, staircase=True)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.0001).minimize(entropy_loss, global_step=global_step)


# Initializer variable
init = tf.global_variables_initializer()


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

inputFile = "./Save/modele/model_" + str(patch_size) + "_" + str(n_filters) + ".ckpt" 
#inputFile = "./Save/modele/model_gaussian_New" + str(patch_size) + "_" + str(n_filters) + ".ckpt"
outputFile = "./Save/modele/model_delete" + str(patch_size) + "_" + str(n_filters) + ".ckpt"  


# GPU and CPU Options
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.905)
device_count={'GPU':1, 'CPU':CPU}
visible_device_list= '0'
config = tf.ConfigProto(device_count=device_count, gpu_options=gpu_options, log_device_placement=True)
config.gpu_options.visible_device_list=visible_device_list
config.gpu_options.allow_growth=True


print('------------------------------------------------------------------------------------------')
print('--------------------------------  starting Training   ------------------------------------')
print('------------------------------------------------------------------------------------------')



loss_min = 2.0
val_acc_max = 0.4
IoUmax = 0.05


with tf.Session() as sess:
    
    # Initialization of variables
    sess.run(init)
    
    # Restore variables from disk.
    saver.restore(sess, inputFile)
    

    print('----------------------------------------------------------------------------------------------')
    print('---------------------------------      Graph Initialized -------------------------------------')
    print('---------------------------------    Iterations started --------------------------------------')
    print('----------------------------------------------------------------------------------------------')
    
    start = time.time()
    for i in range(4):
        
        #Getting the current mini batch

        img_patch_iter, img_mask_iter = sess.run([img_patch_train, img_mask_train])
        
        
        feed = {patch: img_patch_iter, mask: img_mask_iter}

        optimizer_none, loss_iter[i], accuracy_iter[i], IoU_iter[i] = sess.run([optimizer, loss, accuracy, IoU], feed_dict=feed)
        
        Loss, Yout = sess.run([loss, output] , feed_dict=feed)
        
        #print('-----------  Iteration ' + str(i) + '    Loss = ' + str(np.around(loss_iter[i], decimals = decimals)) + '    Accuracy = ' + str(np.around(accuracy_iter[i], decimals = decimals)))
         
        
        if i%step_per_epok == 0 and i> 2:
            
            # Saving loss, IoU and accuracy to disk
            scipy.io.savemat('./Save/plot/loss.mat', mdict={'loss': loss_iter})
            scipy.io.savemat('./Save/plot/accuracy.mat', mdict={'accuracy': accuracy_iter})
            scipy.io.savemat('./Save/plot/iou.mat', mdict={'iou': IoU_iter})
            saver.save(sess, outputFile)
            
            # Calculation validation metrics

            for k in range(step_per_epoch_val):
                
                img_patch_val_iter, img_mask_val_iter = sess.run([img_patch_val, img_mask_val])
                feed_val = {patch: img_patch_val_iter, mask: img_mask_val_iter}
                loss_val[k], accuracy_val[k], IoU_val[k], Y = sess.run([loss, accuracy, IoU, output], feed_dict=feed_val)

            
            end = time.time()
            duration = 1.0*(end-start)/60
            print('------------------------------------------------------------------------------------------------------------------------------------')
            print('----------------  Epoch: ' + str(i//step_per_epok) + '    Mean Loss = ' + str(np.around(np.mean(loss_iter[i-step_per_epok:i]), decimals = decimals)) + ' --------    Mean Accuracy = ' + str(np.around(np.mean(accuracy_iter[i-step_per_epok:i]), decimals=decimals)) + '      Mean IoU = '+ str(np.around(np.mean(IoU_iter[i-step_per_epok:i]), decimals = decimals)) + '    Elapsed Time = ' + str(np.around(duration, decimals = decimals)) + ' mins')
            
            print('----------------              Valid Loss = ' + str(np.around(np.mean(loss_val), decimals = decimals))  + ' --------   Valid Accuracy = ' + str(np.around(np.mean(accuracy_val), decimals = decimals)) + '     Valid IoU = ' + str(np.around(np.mean(IoU_val), decimals = decimals)))
            print('------------------------------------------------------------------------------------------------------------------------------------')


            
            if np.mean(loss_val) <= loss_min:
                saver.save(sess, outputFile)
                loss_min = np.mean(loss_val)
                
                print('------------------------------------------------------------------------------------------------------------------------------------')
                print('------------------------------------------------------------------------------------------------------------------------------------')
                print('------------------------------ ***** Saved variables to Disk Improved LOss *****-------------------------------------------------------------')
                print('------------------------------------------------------------------------------------------------------------------------------------')
                print('------------------------------------------------------------------------------------------------------------------------------------')                
                
                
            elif np.mean(accuracy_val) > val_acc_max:
                saver.save(sess, outputFile)
                val_acc_max = np.mean(accuracy_val)
                
                print('------------------------------------------------------------------------------------------------------------------------------------')
                print('------------------------------------------------------------------------------------------------------------------------------------')
                print('------------------------------ ***** Saved variables to Disk Improved Acc *****-------------------------------------------------------------')
                print('------------------------------------------------------------------------------------------------------------------------------------')
                print('------------------------------------------------------------------------------------------------------------------------------------')
            
#            elif np.mean(IoU_val) > IoUmax:
#                saver.save(sess, outputFile)
#                IoUmax = np.mean(IoU_val)
#                
#                print('------------------------------------------------------------------------------------------------------------------------------------')
#                print('------------------------------------------------------------------------------------------------------------------------------------')
#                print('------------------------------ ***** Saved variables to Disk Improved IoU*****-------------------------------------------------------------')
#                print('------------------------------------------------------------------------------------------------------------------------------------')
#                print('------------------------------------------------------------------------------------------------------------------------------------')            
            
            start = time.time()   
            
    # Training is finished                map(lambda x: 0 if x == 0 else 255, current_mask)
    scipy.io.savemat('./Save/plot/accuracy_Lux.mat', mdict={'accuracy': accuracy_iter})
    scipy.io.savemat('./Save/plot/loss_Lux.mat', mdict={'loss': loss_iter})
    scipy.io.savemat('./Save/plot/iou_Lux.mat', mdict={'accuracy': IoU_iter})
    
    if np.mean(loss_iter[i-step_per_epok:i]) <= loss_min:
        saver.save(sess, outputFile)
        print('------------------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------ ***** Saved variables to Disk *****------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------------------------------------')

print('------------------------------------------------------------------------------------------')
print('--------------------  Each Mini Batch Featrs of shape ' + str(np.shape(img_patch_iter)) + '--------------------')
print('--------------------  Each Mini Batch Labels of shape ' + str(np.shape(img_mask_iter)) + '--------------------')

print('------------------------------------------------------------------------------------------')
print('-------------------------------  Plotting some data   ------------------------------------')
print('------------------------------------------------------------------------------------------')




print('------------------------------------------------------------------------------------------')
print('--------------------------------   Finish     --------------------------------------------')
print('------------------------------------------------------------------------------------------')



seuil = 0.1
Y = Yout
img_patch_val_iter, img_mask_val_iter = img_patch_iter, img_mask_iter
s = 7
preds = (Y > seuil).astype(np.float32)
A = img_patch_val_iter[s, :, :, :]
B = img_mask_val_iter[s, :, :, 0]
C = preds[s, :, :, 0]
D = Y[s, :, :, 0]
plot_sample(A, B, C, D, "Raw Image", 'Ground Truth', 'Prediction', 'Probability Map')
print('Maximum GT = ', np.max(B))


s = s+2
preds = (Y > seuil).astype(np.float32)
A = img_patch_val_iter[s, :, :, :]
B = img_mask_val_iter[s, :, :, 0]
C = preds[s, :, :, 0]
D = Y[s, :, :, 0]
plot_sample(A, B, C, D, "Raw Image", 'Ground Truth', 'Prediction', 'Probability Map')
print('Maximum GT = ', np.max(B))


s = s+3
preds = (Y > seuil).astype(np.float32)
A = img_patch_val_iter[s, :, :, :]
B = img_mask_val_iter[s, :, :, 0]
C = preds[s, :, :, 0]
D = Y[s, :, :, 0]
plot_sample(A, B, C, D, "Raw Image", 'Ground Truth', 'Prediction', 'Probability Map')
print('Maximum GT = ', np.max(B))

s = s+10
preds = (Y > seuil).astype(np.float32)
A = img_patch_val_iter[s, :, :, :]
B = img_mask_val_iter[s, :, :, 0]
C = preds[s, :, :, 0]
D = Y[s, :, :, 0]
plot_sample(A, B, C, D, "Raw Image", 'Ground Truth', 'Prediction', 'Probability Map')
print('Maximum GT = ', np.max(B))

s = s+3
preds = (Y > seuil).astype(np.float32)
A = img_patch_val_iter[s, :, :, :]
B = img_mask_val_iter[s, :, :, 0]
C = preds[s, :, :, 0]
D = Y[s, :, :, 0]
plot_sample(A, B, C, D, "Raw Image", 'Ground Truth', 'Prediction', 'Probability Map')
print('Maximum GT = ', np.max(B))



#s = s+5
#
## Training display
#Patch = img_patch_val_iter[s, :, :, :]
#Mask = img_mask_val_iter[s, :, :, 0]
#
#
## Gaussian only
#gauss = gaussian(Mask, sigma=1.0)
#
## Dilatation only
#dilat= dilation(Mask, selem)
#
## Gauss dilat
#dilat_gauss = dilation(gauss)
#
#
#
#plot_sample(Patch, Mask, gauss, dilat, "Patch", 'Mask', 'Gaussian Dilation', 'Morphological Dilation')
#acc = sum(sum(B==C))/(256*256)
#print(acc)




