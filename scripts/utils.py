"""
Utils including image featurization methods and bootstrap uncertainty for model metrics. 
"""

# general libraries
import numpy as np
from random import choices
import matplotlib.pyplot as plt
from typing import List

# image transformation libraries
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import exposure

# modeling libraries
from transformers import ViTFeatureExtractor, ViTModel
from transformers import BeitFeatureExtractor, BeitModel
from transformers import AutoFeatureExtractor, ResNetModel
import torch


def pca_features(train_images, val_images, test_images, n_dims_kept):

    train_data = np.zeros((11135, 128*128))
    for itr, img in enumerate(train_images):
        train_data[itr,:] = np.mean(img[:,:,:], axis=2).flatten()
    val_data = np.zeros((2088, 128*128))
    for itr, img in enumerate(val_images):
        val_data[itr,:] = np.mean(img[:,:,:], axis=2).flatten()
    test_data = np.zeros((2586, 128*128))
    for itr, img in enumerate(test_images):
        test_data[itr,:] = np.mean(img[:,:,:], axis=2).flatten()

    pca = PCA(n_dims_kept)
    pca.fit(train_data)

    # inspect cumulative explained variance
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    plt.figure()

    # transform the train, val, and test sets into PCA space
    transformed_train = pca.transform(train_data)
    transformed_val = pca.transform(val_data)
    transformed_test = pca.transform(test_data)

    return transformed_train, transformed_val, transformed_test, pca


def HF_last_hidden_state(train_images, val_images, test_images, model_path):
    
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    
    # support for following three models:
    if model_path == 'microsoft/beit-base-patch16-224-pt22k-ft22k':
        feature_extractor = BeitFeatureExtractor.from_pretrained(model_path)
        model = BeitModel.from_pretrained(model_path).to(device)
    elif model_path == 'google/vit-base-patch16-384':
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
        model = ViTModel.from_pretrained(model_path).to(device)
    elif model_path == 'microsoft/resnet-50':
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = ResNetModel.from_pretrained(model_path).to(device)

    train_ViTs = np.zeros((len(train_images), 768))
    for itr, img in enumerate(train_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        last_hidden_states = outputs.last_hidden_state
        if model_path != 'microsoft/resnet-50':
            embedding = np.mean(last_hidden_states[0].cpu().numpy(), axis=0)
        else:
            embedding = last_hidden_states[0].cpu().numpy().flatten()
        train_ViTs[itr, :] = embedding

    val_ViTs = np.zeros((len(val_images), 768))
    for itr, img in enumerate(val_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        last_hidden_states = outputs.last_hidden_state
        if model_path != 'microsoft/resnet-50':
            embedding = np.mean(last_hidden_states[0].cpu().numpy(), axis=0)
        else:
            embedding = last_hidden_states[0].cpu().numpy().flatten()
        val_ViTs[itr, :] = embedding

    test_ViTs = np.zeros((len(test_images), 768))
    for itr, img in enumerate(test_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        last_hidden_states = outputs.last_hidden_state
        if model_path != 'microsoft/resnet-50':
            embedding = np.mean(last_hidden_states[0].cpu().numpy(), axis=0)
        else:
            embedding = last_hidden_states[0].cpu().numpy().flatten()
        test_ViTs[itr, :] = embedding    

    return train_ViTs, val_ViTs, test_ViTs 
    
    
def hog_features(train_images, val_images, test_images):

    train_HOGs = np.zeros((len(train_images), 128, 128))
    for itr, img in enumerate(train_images):
        image = np.dot(img[:,:,:3], [0.2989, 0.5870, 0.1140])
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        train_HOGs[itr, :, :] = hog_image_rescaled

    val_HOGs = np.zeros((len(val_images), 128, 128))
    for itr, img in enumerate(val_images):
        image = np.dot(img[:,:,:3], [0.2989, 0.5870, 0.1140])
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        val_HOGs[itr, :, :] = hog_image_rescaled

    test_HOGs = np.zeros((len(test_images), 128, 128))
    for itr, img in enumerate(test_images):
        image = np.dot(img[:,:,:3], [0.2989, 0.5870, 0.1140])
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        test_HOGs[itr, :, :] = hog_image_rescaled

    return train_HOGs, val_HOGs, test_HOGs 


def nir_features(train_images, val_images, test_images):

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


def bootstrap(predictions: List[int], B: int = 10000, confidence_level: int = 0.95) -> int:

    """
    helper function for providing confidence intervals for sentiment tool
    """

    # compute lower and upper significance index
    critical_value = (1-confidence_level)/2
    lower_sig = 100*critical_value
    upper_sig = 100*(1-critical_value)
    data = []
    for p in predictions:
        data.append(p)

    avgs = []
    # bootstrap resampling loop
    for b in range(B):
        choice = choices(data, k=len(data))
        choice = np.array(choice)
        inner_avg = np.mean(choice)

        avgs.append(inner_avg)

    percentiles = np.percentile(avgs, [lower_sig, 50, upper_sig])

    lower = percentiles[0]
    # median = percentiles[1]
    upper = percentiles[2]

    e_bar = ((np.mean(predictions) - lower) + (upper - np.mean(predictions)))/2
    return e_bar
