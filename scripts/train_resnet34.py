"""
Script to train a Resnet34 with MixUp on the chesapeake bay dataset
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import Add, GlobalAveragePooling2D
from keras import backend as K
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


label_dict = {
    7: 'NaN',
    0: 'Water',#: All areas of open water including ponds, rivers, and lakes
    1: 'Tree Canopy and Shrubs',#: All woody vegetation including trees and shrubs
    2: 'Low Vegetation',#: Plant material less than 2 meters in height including lawns
    3: 'Barren',#: Areas devoid of vegetation consisting of natural earthen material
    4: 'Impervious Surfaces',#: Human-constructed surfaces less than 2 meters in height
    5: 'Impervious Roads',#: Impervious surfaces that are used for transportation
    6: 'Aberdeen Proving Ground',#: U.S. Army facility with no labels

}

label_dict_inv = {v: k for k, v in label_dict.items()}
label_dict_inv

# Data input

train_labels = []
train_images = []
 
for state in ['train_set']:

    fdir_de = glob.glob(f'data/patches_dataset/{state}/**/**/*npy')

    # Combine rgb and nir bands into ful 4-band image

    i_itr = 0
    for f in fdir_de:
        if f.split('.')[0][-3:] == 'nir':
            id_itr = f.split('_')[2].split('/')[-1]
            rgb_f = f.replace("nir", "rgb")
            nir_f = copy.deepcopy(f)
            label = label_dict_inv[f.split('/')[-3]]
            rgb = np.load(rgb_f)
            nir = np.load(nir_f)
            image = np.dstack((rgb, nir))

            train_labels.append(label)
            train_images.append(image)

            i_itr += 1

# Data input

val_labels = []
val_images = []
 
for state in ['val_set']:

    fdir_de = glob.glob(f'data/patches_dataset/{state}/**/**/*npy')

    # Combine rgb and nir bands into ful 4-band image

    i_itr = 0
    for f in fdir_de:
        if f.split('.')[0][-3:] == 'nir':
            id_itr = f.split('_')[2].split('/')[-1]
            rgb_f = f.replace("nir", "rgb")
            nir_f = copy.deepcopy(f)
            label = label_dict_inv[f.split('/')[-3]]
            rgb = np.load(rgb_f)
            nir = np.load(nir_f)
            image = np.dstack((rgb, nir))

            val_labels.append(label)
            val_images.append(image)

            i_itr += 1

# test

# Data input

test_labels = []
test_images = []
 
for state in ['test_set']:

    fdir_de = glob.glob(f'data/patches_dataset/{state}/**/**/*npy')

    # Combine rgb and nir bands into ful 4-band image

    i_itr = 0
    for f in fdir_de:
        if f.split('.')[0][-3:] == 'nir':
            id_itr = f.split('_')[2].split('/')[-1]
            rgb_f = f.replace("nir", "rgb")
            nir_f = copy.deepcopy(f)
            label = label_dict_inv[f.split('/')[-3]]
            rgb = np.load(rgb_f)
            nir = np.load(nir_f)
            image = np.dstack((rgb, nir))

            test_labels.append(label)
            test_images.append(image)

            i_itr += 1

print(len(train_labels))
print(len(train_images))
print(len(val_labels))
print(len(val_images))
print(len(test_labels))
print(len(test_images))

# plot some random images

id_plot = np.random.randint(0, high=10000, size=9)
plt.figure(figsize=(10, 10))

for i, ids in enumerate(id_plot):
    sample_image, sample_label = train_images[ids][:,:,0:3].astype(np.float32), train_labels[ids] 
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(sample_image)
    plt.title(f'Label: {label_dict[sample_label]}')
    plt.axis("off")

# plot some random images

id_plot = np.random.randint(0, high=2000, size=9)
plt.figure(figsize=(10, 10))

for i, ids in enumerate(id_plot):
    sample_image, sample_label = val_images[ids][:,:,0:3].astype(np.float32), val_labels[ids] 
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(sample_image)
    plt.title(f'Label: {label_dict[sample_label]}')
    plt.axis("off")

# plot some random images

id_plot = np.random.randint(0, high=2000, size=9)
plt.figure(figsize=(10, 10))

for i, ids in enumerate(id_plot):
    sample_image, sample_label = test_images[ids][:,:,0:3].astype(np.float32), test_labels[ids] 
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(sample_image)
    plt.title(f'Label: {label_dict[sample_label]}')
    plt.axis("off")

# https://keras.io/examples/vision/mixup/

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128

train_labels = tf.one_hot(np.array(train_labels).astype(np.uint8), 5)
val_labels = tf.one_hot(np.array(val_labels).astype(np.uint8), 5)

train_ds_one = (
    tf.data.Dataset.from_tensor_slices((np.array(train_images).astype(np.float32), train_labels))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((np.array(train_images).astype(np.float32), train_labels))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
)

# Because we will be mixing up the images and their corresponding labels, we will be
# combining two shuffled datasets from the same training data.

train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

val_ds = tf.data.Dataset.from_tensor_slices((np.array(val_images).astype(np.float32), val_labels)).batch(BATCH_SIZE)

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)

# First create the new dataset using our `mix_up` utility
train_ds_mu = train_ds.map(
    lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
)

# Let's preview 9 samples from the dataset
sample_images, sample_labels = next(iter(train_ds_mu))
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
    #ax = plt.subplot(3, 3, i + 1)
    #plt.imshow(image.numpy().squeeze())
    print(label.numpy().tolist())
    #plt.axis("off")

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ResNet 34 (2015)
# Paper: https://arxiv.org/pdf/1512.03385.pdf

n_classes = 5

def stem(inputs):
    """ Construct the Stem Convolution Group
        inputs : input vector
    """
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer="he_normal")(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x
    
def learner(x):
    """ Construct the Learner
        x  : input to the learner
    """
    # First Residual Block Group of 64 filters
    x = residual_group(x, 64, 3)

    # Second Residual Block Group of 128 filters
    x = residual_group(x, 128, 3)

    # Third Residual Block Group of 256 filters
    x = residual_group(x, 256, 5)

    # Fourth Residual Block Group of 512 filters
    x = residual_group(x, 512, 2, False)
    return x

    
def residual_group(x, n_filters, n_blocks, conv=True):
    """ Construct a Residual Group
        x        : input to the group
        n_filters: number of filters
        n_blocks : number of blocks in the group
        conv     : flag to include the convolution block connector
    """
    for _ in range(n_blocks):
        x = residual_block(x, n_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        x = conv_block(x, n_filters * 2)
    return x

def residual_block(x, n_filters):
    """ Construct a Residual Block of Convolutions
        x        : input into the block
        n_filters: number of filters
    """
    shortcut = x
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                      activation="relu", kernel_initializer="he_normal")(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                      activation="relu", kernel_initializer="he_normal")(x)
    x = Add()([shortcut, x])
    return x

def conv_block(x, n_filters):
    """ Construct Block of Convolutions without Pooling
        x        : input into the block
        n_filters: number of filters
    """
    x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same",
                  activation="relu", kernel_initializer="he_normal")(x)
    x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same",
                  activation="relu", kernel_initializer="he_normal")(x)
    return x
    
def classifier(x, n_classes):
    """ Construct the Classifier Group
        x         : input vector
        n_classes : number of output classes
    """
    # Pool at the end of all the convolutional residual blocks
    x = GlobalAveragePooling2D()(x)

    # Final Dense Outputting Layer for the outputs
    outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return outputs

# The input tensor
inputs = Input(shape=(128, 128, 4))

# The Stem Convolution Group
x = stem(inputs)

# The learner
x = learner(x)
    
# The Classifier for n_classes classes
outputs = classifier(x, n_classes)

# Instantiate the Model
model = Model(inputs, outputs)
model.summary()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["categorical_accuracy", f1_score])

es = keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', verbose=1, patience=5)

checkpoint_filepath = 'model_files/resnet34/nov_5_mod2_ckpt'

mc = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_f1_score',
    mode='max',
    save_best_only=True)
history = model.fit(train_ds_mu,
          validation_data=val_ds,
          epochs=25,
          callbacks=[es, mc],
    )

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'], 'o')
plt.plot(history.history['val_categorical_accuracy'], 'o')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], 'o')
plt.plot(history.history['val_loss'], 'o')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
test_ds
# loop through val set, save ground truths and labels

#model = keras.models.load_model()
checkpoint_filepath = 'model_files/resnet34/nov_5_mod2_ckpt'

model.load_weights(checkpoint_filepath)

prediction_full = model.predict(test_ds)
test_predictions = np.array(prediction_full.argmax(axis=-1))
test_predictions
test_labels = np.array(test_labels)
test_labels

print(accuracy_score(test_labels, test_predictions))
print(balanced_accuracy_score(test_labels, test_predictions))
print(f1_score(test_labels, test_predictions, average='macro'))

cm = confusion_matrix(test_labels, test_predictions)
cm