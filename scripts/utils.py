"""
Utils including image featurization methods and bootstrap uncertainty for model metrics. 
"""

# general libraries
import numpy as np
from random import choices

# image transformation libraries
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import exposure

# modeling libraries
from transformers import ViTFeatureExtractor, ViTModel
import torch


def pca(train_images, val_images, test_images, n_dims_kept):

    train_data = np.zeros((11135, 128*128))
    for itr, img in enumerate(train_images):
        train_data[itr,:] = (np.dot(img[:,:,0:3], [0.2989, 0.5870, 0.1140])).flatten()
    val_data = np.zeros((2088, 128*128))
    for itr, img in enumerate(val_images):
        val_data[itr,:] = (np.dot(img[:,:,0:3], [0.2989, 0.5870, 0.1140])).flatten()
    test_data = np.zeros((2586, 128*128))
    for itr, img in enumerate(test_images):
        test_data[itr,:] = (np.dot(img[:,:,0:3], [0.2989, 0.5870, 0.1140])).flatten()

    pca = PCA(n_dims_kept)
    pca.fit(train_data)

    # transform the train, val, and test sets into PCA space
    transformed_train = pca.transform(train_data)
    transformed_val = pca.transform(val_data)
    transformed_test = pca.transform(test_data)

    return transformed_train, transformed_val, transformed_test, pca


def vit_avg_hidden_states(train_images, val_images, test_images):

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True).to(device)
    train_ViTs = np.zeros((len(train_images), 768))
    for itr, img in enumerate(train_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        hidden_states = outputs[2]
        #print(len(hidden_states))  # 13
        #embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]

        att_states_loc = []
        for state in attention_hidden_states:
            att_states_loc.append(state.cpu().numpy())
        att_states_loc = np.array(att_states_loc)

        # take means along features and layers
        att_states_loc = np.mean(att_states_loc, axis=0)
        embedding = np.mean(att_states_loc, axis=1)

        train_ViTs[itr, :] = embedding

    val_ViTs = np.zeros((len(val_images), 768))
    for itr, img in enumerate(val_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        hidden_states = outputs[2]
        #print(len(hidden_states))  # 13
        #embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]

        att_states_loc = []
        for state in attention_hidden_states:
            att_states_loc.append(state.cpu().numpy())
        att_states_loc = np.array(att_states_loc)

        # take means along features and layers
        att_states_loc = np.mean(att_states_loc, axis=0)
        embedding = np.mean(att_states_loc, axis=1)
        val_ViTs[itr, :] = embedding

    test_ViTs = np.zeros((len(test_images), 768))
    for itr, img in enumerate(test_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        hidden_states = outputs[2]
        #print(len(hidden_states))  # 13
        #embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]

        att_states_loc = []
        for state in attention_hidden_states:
            att_states_loc.append(state.cpu().numpy())
        att_states_loc = np.array(att_states_loc)

        # take means along features and layers
        att_states_loc = np.mean(att_states_loc, axis=0)
        embedding = np.mean(att_states_loc, axis=1)
        test_ViTs[itr, :] = embedding    

    return train_ViTs, val_ViTs, test_ViTs 


def vit_last_hidden_state(train_images, val_images, test_images):
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)

    train_ViTs = np.zeros((len(train_images), 768))
    for itr, img in enumerate(train_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        last_hidden_states = outputs.last_hidden_state
        embedding = np.mean(last_hidden_states[0].cpu().numpy(), axis=0)
        train_ViTs[itr, :] = embedding

    val_ViTs = np.zeros((len(val_images), 768))
    for itr, img in enumerate(val_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        last_hidden_states = outputs.last_hidden_state
        embedding = np.mean(last_hidden_states[0].cpu().numpy(), axis=0)
        val_ViTs[itr, :] = embedding

    test_ViTs = np.zeros((len(test_images), 768))
    for itr, img in enumerate(test_images):
        image = img[:,:,0:3]
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        last_hidden_states = outputs.last_hidden_state
        embedding = np.mean(last_hidden_states[0].cpu().numpy(), axis=0)
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


def bootstrap(gold, predictions, metric, B=10000, confidence_level=0.95):
    critical_value=(1-confidence_level)/2
    lower_sig=100*critical_value
    upper_sig=100*(1-critical_value)
    data=[]
    for g, p in zip(gold, predictions):
        data.append([g,p])

    accuracies=[]
    
    for b in range(B):
        choice=choices(data, k=len(data))
        choice=np.array(choice)
        accuracy=metric(choice[:,0], choice[:,1])
        
        accuracies.append(accuracy)
    
    percentiles=np.percentile(accuracies, [lower_sig, 50, upper_sig])
    
    lower=percentiles[0]
    median=percentiles[1]
    upper=percentiles[2]
    
    return lower, median, upper