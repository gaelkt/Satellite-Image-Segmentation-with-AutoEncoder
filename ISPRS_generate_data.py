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
    print('Shape of TIF LABEL IS  ', np.shape(image_label), 'Maximum is, ', np.max(image_label))
    
    x_row = [c for c in range(x_I) if c%stride ==0 and c + size_x <= x_I]
    y_col = [c for c in range(y_I) if c%stride ==0 and c + size_y <= y_I]
    samples_labels = [image_label[x:x+size_x, y:y+size_y] for x in x_row for y in y_col]
    samples_labels = np.transpose(samples_labels, (0, 1, 2))
    samples_labels[samples_labels > 255] = 255
    return samples_labels


# Divide de features image of 5000x5000x3 to several patches of resolution size_x x size_y with a step of stride 
def sampling_features(filename, size_x = 256, size_y = 256, stride = 256):
    image_gdal = gdal.Open(filename, gdal.GA_ReadOnly)
    image_features = image_gdal.ReadAsArray()
    image_features = np.transpose(image_features, (1, 2, 0))
    #print(np.shape(image_features))
    x_I, y_I, N_channel = np.shape(image_features)
    print('Shape of TIF Image IS  ', np.shape(image_features), 'Maximum is, ', np.max(image_features))
    x_row = [c for c in range(x_I) if c%stride ==0 and c + size_x <= x_I]
    y_col = [c for c in range(y_I) if c%stride ==0 and c + size_y <= y_I]
    features = [[image_features[x:x+size_x, y:y+size_y, channel] for x in x_row for y in y_col] for channel in range(N_channel)]
    features = np.transpose(features, (1, 2, 3, 0))
    features[features > 255] = 255
    del image_features, image_gdal
    return features


# Generate a dataset of images of  [size_x, size_y, channel]
def generate_dataset(principal_folder, folder_features_sampled, folder_labels_sampled, size_x = 256, size_y = 256, stride = 256, is_Training = True, folder_average = ''):
    batch_features = []
    batch_labels = []
    i, k = 0, 0
    # Image that will store average values for pixels
    AVERAGE = np.zeros((size_x, size_y, 3))
    
    
    cities = os.listdir(principal_folder)
    
    for city in cities:
        filename_labels = principal_folder  +  str(city) + "/" + str(city) + "_Roads.tif"
        print(filename_labels)
        filename_features = principal_folder  +  str(city) + "/" + str(city) + "_Image.tif"
        print(filename_features)
    
            
        # We add the samples coming from the current image file to the current bqtch
        batch_features.extend(sampling_features(filename_features, size_x, size_y , stride))
        batch_labels.extend(sampling_labels(filename_labels, size_x, size_y , stride))
        
        
        
        print('Fichier TIF ' +  str(i) + '   city = ' + str(city))
        
        print(len(batch_features))
        
            
        for j in range(len(batch_features)):
            #Saving patch j
           
            
            data = batch_features[0]
            datamax = np.max(data)
            if datamax > 255:
                print(np.max(data), np.shape(data))
            img_features = Image.fromarray(batch_features[0].astype(np.uint8))
            filename_patch = folder_features_sampled + str(city) + '_' + str(i) + '_' + str(j) + '.jpeg'

            img_features.save(filename_patch)
            
            # We add this patch to the calculation of average data
            if is_Training:
                AVERAGE = AVERAGE + batch_features[0]
            
            # Removing patch j from the queue
            batch_features.pop(0)
            
            
            # Saving masl j
            img_labels = Image.fromarray(batch_labels[0])
            filename_mask = folder_labels_sampled +  str(city) + '_' + str(i) + '_' + str(j) + '.jpeg'
            
            # 'LA' is to convert the image into grayscale
            img_labels = img_labels.convert("L")
            img_labels.save(filename_mask)
            
            # Removing mask j from the queue
            batch_labels.pop(0)
            
            k = k+1
        i = i+1
        
    # At the end we save the average image
    if is_Training:
        AVERAGE = 1.0*AVERAGE/k
        AVERAGE_Image = Image.fromarray(AVERAGE.astype(np.uint8))
        filename_Average= folder_average + 'average.jpeg'
        AVERAGE_Image.save(filename_Average)
            
    print('Dataset Generated with '+ str(k) + ' patches and ' + str(i) + '  images')
            

####  Location of TIF images         

# LOcation of images   
principal_folder_tif_train = "../../../dataset_ISPRS/TIF_Images_Train/"

principal_folder_tif_valid = "../../../dataset_ISPRS/TIF_Images_Valid/"

# Average Image    
folder_average = "../../../../dataset_ISPRS/Average/"


# Validation Data    
#folder_features_val, folder_labels_val = "../../../Documents/GeoAzur/dataset/AerialImageDataset/val/images/", "../../../Documents/GeoAzur/dataset/AerialImageDataset/val/gt/"

### Location to store sampled JPEG images

# Training Data   
folder_features_sampled_train, folder_labels_sampled_train = "../../../dataset_ISPRS/sampled/train/patch/", "../../../dataset_ISPRS/sampled/train/mask/"

# Validation Data   
folder_features_sampled_valid, folder_labels_sampled_valid = "../../../dataset_ISPRS/sampled/valid/patch/", "../../../dataset_ISPRS/sampled/valid/mask/"


list_cities_train = os.listdir(principal_folder_tif_train)
list_cities_valid = os.listdir(principal_folder_tif_valid)

print('Training cities are ' + str(len(list_cities_train)))
print('Validation cities images are ' + str(len(list_cities_valid)))

## Uncomment this to generate the training dataset
generate_dataset(principal_folder_tif_train, folder_features_sampled_train, folder_labels_sampled_train, size_x = 256, size_y = 256, stride = 256)

## Uncomment this to generate the validation dataset
generate_dataset(principal_folder_tif_valid, folder_features_sampled_valid, folder_labels_sampled_valid, size_x = 256, size_y = 256, stride = 256)