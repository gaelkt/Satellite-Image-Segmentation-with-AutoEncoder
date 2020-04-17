#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:49:35 2019

@author: gael
"""

from osgeo import gdal
import numpy as np
from PIL import Image
import os


"""
This file takes TIF images, then divide them into small patches of size [size_x, size_y, channel] and
store then in a folder. The patches are jpeg images
We do this for training, validation and testing
We use a sliding window and a patche to achieve this operation
size_x = height of the sliding window
size_y = width of the sliding window
stride = stride during the operation


"""



# Divide de ground truth TIF image of 5000x5000 into several patches of resolution size_x x size_y
def sampling_labels(filename, size_x = 256, size_y = 256, stride = 256):
    image_gdal = gdal.Open(filename, gdal.GA_ReadOnly)
    image_label = image_gdal.ReadAsArray()
    x_I, y_I = np.shape(image_label)
    #print('Shape of TIF LABEL IS  ', np.shape(image_label), 'Maxinun is, ', np.max(image_label))
    
    x_row = [c for c in range(x_I) if c%stride ==0 and c + size_x <= x_I]
    y_col = [c for c in range(y_I) if c%stride ==0 and c + size_y <= y_I]
    samples_labels = [image_label[x:x+size_x, y:y+size_y] for x in x_row for y in y_col]
    samples_labels = np.transpose(samples_labels, (0, 1, 2))
    return samples_labels 


# Divide de features image of 5000x5000x3 to several patches of resolution size_x x size_y with a step of stride 
def sampling_features(filename, size_x = 256, size_y = 256, stride = 256):
    image_gdal = gdal.Open(filename, gdal.GA_ReadOnly)
    image_features = image_gdal.ReadAsArray()
    image_features = np.transpose(image_features, (1, 2, 0))
    #print(np.shape(image_features))
    x_I, y_I, N_channel = np.shape(image_features)
    x_row = [c for c in range(x_I) if c%stride ==0 and c + size_x <= x_I]
    y_col = [c for c in range(y_I) if c%stride ==0 and c + size_y <= y_I]
    features = [[image_features[x:x+size_x, y:y+size_y, channel] for x in x_row for y in y_col] for channel in range(N_channel)]
    features = np.transpose(features, (1, 2, 3, 0))
    del image_features, image_gdal
    return features


# Generate a dataset of images of  [size_x, size_y, channel]
def generate_dataset(list_name, folder_labels, folder_features, folder_features_sampled, folder_labels_sampled, size_x = 256, size_y = 256, stride = 256, is_Training = True, folder_average = ''):
    batch_features = []
    batch_labels = []
    i, k = 0, 0
    # Image that will store average values for pixels
    AVERAGE = np.zeros((size_x, size_y, 3))

    for name in list_name:
        # We get each image file and its label
        filename_labels = folder_labels + name
        filename_features = folder_features + name
            
        # We add the samples coming from the current image file to the current bqtch
        batch_features.extend(sampling_features(filename_features, size_x, size_y , stride))
        batch_labels.extend(sampling_labels(filename_labels, size_x, size_y , stride))
        
        
        
        print('Fichier TIF ' +  str(i))
        
            
        for j in range(len(batch_features)):
            #Saving patch j
            img_features = Image.fromarray(batch_features[0])
            filename_feature = folder_features_sampled + 'img_' + str(i) + '_' + str(j) + '.jpeg'
            img_features.save(filename_feature)
            
            # We add this patch to the calculation of average data
            if is_Training:
                AVERAGE = AVERAGE + batch_features[0]
            
            # Removing patch j from the queue
            batch_features.pop(0)
            
            
            # Saving masl j
            img_labels = Image.fromarray(batch_labels[0])
            filename_labels = folder_labels_sampled + 'img_' + str(i) + '_' + str(j) + '.jpeg'
            img_labels.save(filename_labels)
            
            # Removing mask j from the queue
            batch_labels.pop(0)
            
            k = k+1
        i = i+1
        
    # At the end we save the average image
    if is_Training:
        AVERAGE = 1.0*AVERAGE/k
        AVERAGE_Image = Image.fromarray(batch_features[0])
        filename_Average= folder_average + 'average.jpeg'
        AVERAGE_Image.save(filename_Average)
            
    print('Dataset Generated with '+ str(k) + ' patches and ' + str(i) + 'images')
            

####  Location of TIF images         

# Training Data    
folder_features_train, folder_labels_train = "../../../Documents/GeoAzur/dataset/AerialImageDataset/train/images/", "../../../Documents/GeoAzur/dataset/AerialImageDataset/train/gt/"

# Average Image    
folder_average = "../../../Documents/GeoAzur/dataset/AerialImageDataset/"


# Validation Data    
folder_features_val, folder_labels_val = "../../../Documents/GeoAzur/dataset/AerialImageDataset/val/images/", "../../../Documents/GeoAzur/dataset/AerialImageDataset/val/gt/"

### Location to store sampled JPEG images

# Training Data   
folder_features_sampled_train, folder_labels_sampled_train = "../../../Documents/GeoAzur/dataset/AerialImageDataset/sampled/patches/", "../../../Documents/GeoAzur/dataset/AerialImageDataset/sampled/masks/"

# Validation Data   
folder_features_sampled_val, folder_labels_sampled_val = "../../../Documents/GeoAzur/dataset/AerialImageDataset/sampled_val/patches/", "../../../Documents/GeoAzur/dataset/AerialImageDataset/sampled_val/masks/"

Image_Name_List_train = os.listdir(folder_features_train)  # Name of all training images
Image_Name_List_val = os.listdir(folder_features_val)  # Name of all validation images

print('Training images are ' + str(len(Image_Name_List_train)))
print('Validation images are ' + str(len(Image_Name_List_val)))

## Uncomment this to generate the training dataset
#generate_dataset(Image_Name_List_train, folder_labels_train, folder_features_train, folder_features_sampled_train, folder_labels_sampled_train, size_x = 256, size_y = 256, stride = 256)

## Uncomment this to generate the validation dataset
generate_dataset(Image_Name_List_val, folder_labels_val, folder_features_val, folder_features_sampled_val, folder_labels_sampled_val, size_x = 256, size_y = 256, stride = 256)