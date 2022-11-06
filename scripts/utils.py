"""
Utils including image featurization methods and bootstrap uncertainty for model metrics. 
"""

# general libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import copy
from numpy import matlib
import seaborn as sns

# image transformation libraries
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import data, exposure

# modeling libraries
import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
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

    NIRs = []
    NDVIs = []
    NDWIs = []
    for ids in range(len(train_images)):
        NIR = train_images[ids][:,:,3]
        NIRs.append(NIR)
        RED = train_images[ids][:,:,0]
        NDVI = (NIR-RED)/(NIR+RED)
        NDVIs.append(NDVI)
        G = train_images[ids][:,:,1]
        NDWI = (G-NIR)/(G+NIR)    
        NDWIs.append(NDWI)
    
    transformed_train = np.dstack((np.array(NIRs), np.array(NDVIs), np.array(NDWIs)))

    NIRs = []
    NDVIs = []
    NDWIs = []
    for ids in range(len(val_images)):
        NIR = val_images[ids][:,:,3]
        NIRs.append(NIR)
        RED = val_images[ids][:,:,0]
        NDVI = (NIR-RED)/(NIR+RED)
        NDVIs.append(NDVI)
        G = val_images[ids][:,:,1]
        NDWI = (G-NIR)/(G+NIR)    
        NDWIs.append(NDWI)
    
    transformed_val= np.dstack((np.array(NIRs), np.array(NDVIs), np.array(NDWIs)))

    NIRs = []
    NDVIs = []
    NDWIs = []
    for ids in range(len(test_images)):
        NIR = test_images[ids][:,:,3]
        NIRs.append(NIR)
        RED = test_images[ids][:,:,0]
        NDVI = (NIR-RED)/(NIR+RED)
        NDVIs.append(NDVI)
        G = test_images[ids][:,:,1]
        NDWI = (G-NIR)/(G+NIR)    
        NDWIs.append(NDWI)
    
    transformed_test = np.dstack((np.array(NIRs), np.array(NDVIs), np.array(NDWIs)))


    return transformed_train, transformed_val, transformed_test 


#def ViT_features(train_images, val_images, test_images):

#def HOG_features(train_images, val_images, test_images):


