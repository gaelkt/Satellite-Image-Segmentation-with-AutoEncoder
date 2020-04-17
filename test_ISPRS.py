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

from osgeo import gdal
from osgeo import osr
import numpy as np
import shutil


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
#sys.path.insert(0, 'libs/')
sys.path.insert(0, './libs/')
from functions import plot_sample, cross_entropy_loss, soft_intersection_union, accuracy_metric, iou_metric, plot_training
from unet_tf import create_unet
import os
import time
from sklearn.metrics import accuracy_score
import scipy.misc
from skimage.morphology import erosion, dilation, opening, closing, white_tophat

from skimage import data, exposure, img_as_float

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



# Parameters of the model

# Size of each patch
# Each TIF image is divided into patches of size [patch_size, patch_size, nb_channel] with a stride 
patch_size, nb_channel, stride, classes = 3072, 3, 2048, 1   # size of new sampled image and stride for patches


# Unet Parameters
n_filters, kernel_size_down, kernel_size_up = 32, 3, 2

# Training Parameters
batch_size, epoch = 32, 1



# Threshold for decision
threshold = 0.5

# Decimals for printing results
decimals = 3

gamma = 0.5




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
    import scipy.misc
    4. For this purpose, we use tf.Data API
    
    For more info read this:
        https://medium.com/@nimatajbakhsh/building-multi-threaded-custom-data-pipelines-for-tensorflow-f76e9b1a32f5
    
"""




# Set up Multi-threaded Batch generator
            
# Parameters for multi-threading
output_types = (tf.float32, tf.float32)  # Types of generetor outputs 
CPU = multiprocessing.cpu_count()        # NUmber of available CPU on the computer
Thread = 2*CPU-5                                  # There is 2 Threads/CPU and we leave 5 threads unused to avoid overflow
Thread = 10




print('---------------------------         Generator Ok     -------------------------------------')



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


output = create_unet(patch, nb_channel, kernel_size_down, kernel_size_up, n_filters, classes, training = False)




if classes ==1:
    mask_soft_pred = tf.nn.sigmoid(output)
else:
    mask_soft_pred  = ''



if classes > 1:
    """ We have multi class segmentation:
        1. mask_hard_true: True mask hard coded with shape [batch_size, patch_size, patch_size]
           mask is by default soft coded with shape [batch_size, patch_size, patch_size, classes]

        2. mask_hard_true_resh: True mask hard coded and reshaped to a vector with shape [batch_size*patch_size*patch_size]
        
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
    a = [random.randint(0, 2) + random.random() for i in range(5)]
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
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)

## Optimizer with an exponential decay learning rate
#lr = 0.1  
#decay_steps = epoch*step_per_epok
#decay = 0.95
#global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay, staircase=True)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.0001).minimize(entropy_loss, global_step=global_step)


# Initializer variable
init = tf.global_variables_initializer()


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

#inputFile = "./Save/modele/model_" + str(256) + "_" + str(n_filters) + ".ckpt" 
inputFile ="./Save/modele/model_II" + str(256) + "_" + str(n_filters) + ".ckpt" 

# GPU and CPU Options
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.925)
device_count={'GPU':1, 'CPU':CPU}
visible_device_list= '0'
config = tf.ConfigProto(device_count=device_count, gpu_options=gpu_options, log_device_placement=True)
config.gpu_options.visible_device_list=visible_device_list
config.gpu_options.allow_growth=True


print('------------------------------------------------------------------------------------------')
print('--------------------------------  starting Training   ------------------------------------')



loss_min = 0.435
val_acc_max = 0.838
IoUmax = 0.561

city = 'Ville'

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
    
    for i in range(1):
        
        filename = 'Test/'+city+'/'+city+'_Pred_II'
        
        filename_Original = 'Test/'+city+'/'+city+'_original'
        
        filename_New = filename
        
        shutil.copy(filename_Original, filename_New) 

        image_gdal = gdal.Open(filename, gdal.GA_Update)
        image_TIF = image_gdal.ReadAsArray()
        image_TIF = np.transpose(image_TIF, (1, 2, 0))

        

        
    
        print('Shape of TIF Image is ', np.shape(image_TIF), 'Maximum is, ', np.max(image_TIF))
    
        x_I, y_I, N_channel = np.shape(image_TIF)
        print('Shape is ', x_I, y_I, N_channel)
        image_TIF_II = np.zeros((x_I, y_I))
        image_TIF_I0 = np.zeros((x_I, y_I, 3))
    
        x_row = [c for c in range(x_I) if c%stride ==0 and c + patch_size <= x_I]
        y_col = [c for c in range(y_I) if c%stride ==0 and c + patch_size <= y_I]
    
        for x in x_row:
            for y in y_col:
                patch_tif = image_TIF[x:x+patch_size, y:y+patch_size, :]
                
                # Normalization of patches 
                patch_tif = exposure.adjust_gamma(patch_tif, gamma)
            
                patch_tif = patch_tif/255.0
                
                patch_tif = np.expand_dims(patch_tif, axis=0)

                feed = {patch: patch_tif}

                logits = sess.run(mask_soft_pred, feed_dict=feed)
            
                image_TIF_II[x:x+patch_size, y:y+patch_size] = logits[0, :, :, 0]
                
                image_TIF_I0[x:x+patch_size, y:y+patch_size] = image_TIF[x:x+patch_size, y:y+patch_size]
                
            # Borders    col
            patch_tif = image_TIF[x:x+patch_size, y_I-patch_size:y_I, :] 
            patch_tif = patch_tif/255.0
            patch_tif = np.expand_dims(patch_tif, axis=0)
            feed = {patch: patch_tif}
            logits = sess.run(mask_soft_pred, feed_dict=feed)
            image_TIF_II[x:x+patch_size, y_I-patch_size:y_I] = logits[0, :, :, 0]
            image_TIF_I0[x:x+patch_size, y_I-patch_size:y_I] = image_TIF[x:x+patch_size, y_I-patch_size:y_I]
            
        
         # Borders    row
        for y in y_col:
            patch_tif = image_TIF[x_I-patch_size:x_I, y:y+patch_size, :]
            
            patch_tif = patch_tif/255.0
                
            patch_tif = np.expand_dims(patch_tif, axis=0)

            feed = {patch: patch_tif}

            logits = sess.run(mask_soft_pred, feed_dict=feed)
            
            image_TIF_II[x_I-patch_size:x_I, y:y+patch_size] = logits[0, :, :, 0]
                
            image_TIF_I0[x_I-patch_size:x_I, y:y+patch_size] = image_TIF[x_I-patch_size:x_I, y:y+patch_size]


        # Last cell
        patch_tif = image_TIF[x_I-patch_size:x_I, y_I-patch_size:y_I, :]
            
        patch_tif = patch_tif/255.0
                
        patch_tif = np.expand_dims(patch_tif, axis=0)

        feed = {patch: patch_tif}

        logits = sess.run(mask_soft_pred, feed_dict=feed)
            
        image_TIF_II[x_I-patch_size:x_I, y_I-patch_size:y_I] = logits[0, :, :, 0]
        
        image_TIF_I0[x_I-patch_size:x_I, y_I-patch_size:y_I] = image_TIF[x_I-patch_size:x_I, y_I-patch_size:y_I]        
        
        
        
          
        image_TIF_II = image_TIF_II.astype(np.float32)
        
        Prediction = image_TIF_II
        
        
        Prediction = 255*Prediction
        
        Prediction = Prediction.astype(np.uint8)
        
        Prediction[Prediction >= 255] = 255
        
        Prediction[Prediction <= 1] = 1
        
        image_TIF_II = image_TIF_II.astype(np.uint8)
        
   
        #Prediction = closing(Prediction)
        
        image_TIF_RED = 1*Prediction
        
        image_TIF_GRE = 1*Prediction
        
        image_TIF_BLU = 1*Prediction
        
        print('Min of Prediction', np.min(Prediction))
        
        print('Max of Prediction', np.max(Prediction))
                
        #Prediction = closing(Prediction)
        

        

        print('enddddddddd')
        
        ab = image_gdal.GetRasterBand(1)
        
        print('ab')
        
        image_gdal.GetRasterBand(1).WriteArray(image_TIF_RED)
        print('ok1')
        image_gdal.GetRasterBand(2).WriteArray(image_TIF_GRE)
        print('ok2')
        image_gdal.GetRasterBand(3).WriteArray(image_TIF_BLU)
        
        image_gdal.FlushCache()
        print('ok--')
        
        image_gdal = None

