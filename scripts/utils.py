"""
Utils including image featurization methods and bootstrap uncertainty for model metrics. 
"""

# general libraries
import glob
import numpy as np
import os
import numpy as np
import pandas as pd

# image transformation libraries
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import data, exposure

# modeling libraries
from transformers import ViTFeatureExtractor, ViTModel
from datasets import load_dataset
import torch


def PCA_features(train_images, val_images, test_images, n_dims_kept):
    # first grab (n imgs x n pixels) datasets for PCA (from images converted to greyscale)

    train_data = np.zeros((11135, 128*128))
    for itr, img in enumerate(train_images):
        train_data[itr,:] = (np.dot(img[:,:,0:3], [0.2989, 0.5870, 0.1140])).flatten()
    val_data = np.zeros((2088, 128*128))
    for itr, img in enumerate(val_images):
        val_data[itr,:] = (np.dot(img[:,:,0:3], [0.2989, 0.5870, 0.1140])).flatten()
    test_data = np.zeros((2586, 128*128))
    for itr, img in enumerate(test_images):
        test_data[itr,:] = (np.dot(img[:,:,0:3], [0.2989, 0.5870, 0.1140])).flatten()

    # all are in expected shape
    #print(train_data.shape)
    #print(val_data.shape)
    #print(test_data.shape)

    pca = PCA(n_dims_kept)
    pca.fit(train_data)

    # inspect cumulative explained variance
    #plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o')
    #plt.xlabel('number of components')
    #plt.ylabel('cumulative explained variance');

    # transform the train, val, and test sets into PCA space
    transformed_train = pca.transform(train_data)
    transformed_val = pca.transform(val_data)
    transformed_test = pca.transform(test_data)

    return transformed_train, transformed_val, transformed_test



def NIR_features(train_images, val_images, test_images):

    train_NIRs = np.zeros((len(train_images), 128, 128, 3))
    for ids in range(len(train_images)):
        NIR = train_images[ids][:,:,3]
        RED = train_images[ids][:,:,0]
        NDVI = (NIR-RED)/(NIR+RED)
        G = train_images[ids][:,:,1]
        NDWI = (G-NIR)/(G+NIR)    
        train_NIRs[ids, :, :, 0] = NIR
        train_NIRs[ids, :, :, 1] = NDVI
        train_NIRs[ids, :, :, 2] = NDWI

    val_NIRs = np.zeros((len(val_images), 128, 128, 3))
    for ids in range(len(val_images)):
        NIR = val_images[ids][:,:,3]
        RED = val_images[ids][:,:,0]
        NDVI = (NIR-RED)/(NIR+RED)
        G = val_images[ids][:,:,1]
        NDWI = (G-NIR)/(G+NIR)    
        val_NIRs[ids, :, :, 0] = NIR
        val_NIRs[ids, :, :, 1] = NDVI
        val_NIRs[ids, :, :, 2] = NDWI

    test_NIRs = np.zeros((len(test_images), 128, 128, 3))
    for ids in range(len(test_images)):
        NIR = test_images[ids][:,:,3]
        RED = test_images[ids][:,:,0]
        NDVI = (NIR-RED)/(NIR+RED)
        G = test_images[ids][:,:,1]
        NDWI = (G-NIR)/(G+NIR)    
        test_NIRs[ids, :, :, 0] = NIR
        test_NIRs[ids, :, :, 1] = NDVI
        test_NIRs[ids, :, :, 2] = NDWI
    
    return train_NIRs, val_NIRs, test_NIRs 


#def ViT_features(train_images, val_images, test_images):

#def HOG_features(train_images, val_images, test_images):


