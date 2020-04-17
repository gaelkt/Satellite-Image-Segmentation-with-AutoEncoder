#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 12:35:27 2020
This file is used for data augmentation
    1. For each patch, we do vertical flip, horizontal flip and three rotation. In addition we add a gaussian noise tp the patch
    
    2. For each mask, we do the corresponding operations: vertical flip, horozontal flip and rotation. But no noise is added

@author: Gael Kamdem De Teyou for IRD/Geoazur/Luxcarta
"""

from PIL import Image, ImageFilter
import os
import random
import sys

"""
Specify the locations of patch and mask to augment and also the locations for augmented data
"""
patch_folder = "../../../dataset_ISPRS/sampled/train/patch/"

mask_folder = "../../../dataset_ISPRS/sampled/train/mask/"


patch_folder_new = "../../../dataset_ISPRS/sampled/augmented/patch/"

mask_folder_new = "../../../dataset_ISPRS/sampled/augmented/mask/"


def remove_from_string(expression, name):
    
    return name.replace('.jpeg', '')



patch_name = os.listdir(patch_folder)
mask_name = os.listdir(mask_folder)

mask_name_equivalent = [a.replace('mask', 'patch') for a in mask_name]

# Sorting to have the same order for both lists
patch_name.sort() 
mask_name_equivalent.sort() 
mask_name.sort() 

#mismatch = [a for a in patch_name if a not in mask_name]


if patch_name != mask_name_equivalent:
    print('--------------------------------- Different names between patches and images -----------------------')
    print('Nb of patches = ' + str(len(patch_name)))
    print('Nb of masks = ' + str(len(mask_name)))
    sys.exit(" -----------------    There are mismatches between patches and masks ----------------------")
else:
    print('--------------------------  Same names for patches and images -------------------------')
    print('----------------------------There are ' + str(len(patch_name)) + ' patches to augment--------------------')
    print('------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------')
    
    for i in range(len(patch_name)):
        if i%100 == 0:
            print('Fichier ' + str(i) + ' out of ' + str(len(patch_name)))
         
        # radious for gaussian blur    
        #a = [random.randint(0, 2) + random.random() for i in range(5)]
        angles = [90, 180, 270]
        
        # Reading patch and mask
        patch = Image.open(patch_folder + patch_name[i])
        mask = Image.open(mask_folder + mask_name[i])
        
        if patch_name[i] != mask_name[i]:
            print('Errorrrrrrrrrrrrrrrrrrrrrrr')
            sys.exit(" ----- Problem with this file -" + str(patch_name[i]) + ' and ' + str(mask_name[i]))
        
        
#        # We want patch and mask to have the same name
#        mask_name[i] = mask_name[i].replace('mask', 'img')
#        patch_name[i] = patch_name[i].replace('patch', 'img')
        
        # No changes
        patch.save(patch_folder_new + patch_name[i])
        mask.save(mask_folder_new + mask_name[i])
        
        # Flipping patches and masks
        
        # Vertical
        patch_vertical_flip_top_bottom = patch.transpose(Image.FLIP_TOP_BOTTOM)
        #patch_vertical_flip_top_bottom = patch_vertical_flip_top_bottom.filter(ImageFilter.GaussianBlur(a[0]))
        patch_vertical_flip_top_bottom.save(patch_folder_new + remove_from_string('.jpeg', patch_name[i]) + '_ver.jpeg')
        
        mask_vertical_flip_top_bottom = mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask_vertical_flip_top_bottom.save(mask_folder_new + remove_from_string('.jpeg', mask_name[i]) + '_ver.jpeg')
        
        # Horizontal
        patch_horizontal_flip_top_bottom = patch.transpose(Image.FLIP_LEFT_RIGHT)
        #patch_horizontal_flip_top_bottom = patch_horizontal_flip_top_bottom.filter(ImageFilter.GaussianBlur(a[1]))
        patch_horizontal_flip_top_bottom.save(patch_folder_new + remove_from_string('.jpeg', patch_name[i]) + '_hor.jpeg')
        
        mask_horizontal_flip_top_bottom = mask.transpose(Image.FLIP_LEFT_RIGHT)
        mask_horizontal_flip_top_bottom.save(mask_folder_new + remove_from_string('.jpeg', mask_name[i]) + '_hor.jpeg')

        # Rotations
        k = 0
        for angle in angles:
            
            patch_rot = patch.rotate(angle)
            #patch_rot = patch_rot.filter(ImageFilter.GaussianBlur(a[2+k]))
            patch_rot.save(patch_folder_new + remove_from_string('.jpeg', patch_name[i]) + '_rot_'+str(angle) + '_.jpeg')
            
            
            mask_rot = mask.rotate(angle)
            mask_rot.save(mask_folder_new + remove_from_string('.jpeg', mask_name[i]) + '_rot_'+str(angle) + '_.jpeg')
            
            
            patch_rot_vert_flip = patch_vertical_flip_top_bottom.rotate(angle)
            patch_rot_vert_flip.save(patch_folder_new + remove_from_string('.jpeg', patch_name[i]) + '_vert_rot_'+str(angle) + '_.jpeg')
            
            mask_rot_vert_flip = mask_vertical_flip_top_bottom.rotate(angle)
            mask_rot_vert_flip.save(mask_folder_new + remove_from_string('.jpeg', mask_name[i]) + '_vert_rot_'+str(angle) + '_.jpeg')   
            
            
            patch_rot_hor_flip = patch_horizontal_flip_top_bottom.rotate(angle)
            patch_rot_hor_flip.save(patch_folder_new + remove_from_string('.jpeg', patch_name[i]) + '_hori_rot_'+str(angle) + '_.jpeg')     
            
            mask_rot_hori_flip = mask_horizontal_flip_top_bottom.rotate(angle)
            mask_rot_hori_flip.save(mask_folder_new + remove_from_string('.jpeg', mask_name[i]) + '_hori_rot_'+str(angle) + '_.jpeg')     
            
            
            k = k+1

patch_name_new = os.listdir(patch_folder_new)
mask_name_new = os.listdir(mask_folder_new) 

patch_name_new.sort() 
mask_name_new.sort() 

if patch_name_new != mask_name_new:
    sys.exit(" -----------------    There are mismatches between patches and masks ----------------------")
else:
    print('---------------------------- Now There are ' + str(len(patch_name_new)) + ' new patches --------------------')
    print('-----------------------------            Finish       ------------------------------------')
    print('------------------------------------------------------------------------------------------')


