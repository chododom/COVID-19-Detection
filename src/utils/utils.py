"""
This file contains definitions of various helper functions and utilities used during our experimentation.

Author: Dominik Chodounský
Institution: Faculty of Information Technology, Czech Technical University in Prague
Last edit: 2021-05-12
"""


import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import shutil


def show_sample_grid(path, grid_size=6):
    """
    Prints a grid of an equal representation of randomly selected images from the 'negative' class and the 'positive' class.
    
    Parameters
    ----------
    path : str
        Path to folder contaning the data. This folder should contain one subfolder per class (negative and positive) with the image data within them.
    grid_size : int
        Size of the side of the grid, the resulting number of images displayed will be grid_size * grid_size. Default value: 6.
    """
    
    if grid_size % 2 != 0:
        raise ValueError(f'The grid size must be divisible by 2, so that both classes can be represented equally')

    negative = random.sample(os.listdir(os.path.join(path, 'negative')), int(grid_size**2 / 2))
    positive = random.sample(os.listdir(os.path.join(path, 'positive')), int(grid_size**2 / 2))

    negative_imgs = [cv2.imread(os.path.join(path, 'negative', i)) for i in negative]
    positive_imgs = [cv2.imread(os.path.join(path, 'positive', i)) for i in positive]

    plt.figure(figsize=(12,12))
    i, j = 0, 0
    for cnt in range(grid_size**2):
        ax = plt.subplot(grid_size, grid_size, cnt + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if cnt % 2 == 0:
            plt.imshow(negative_imgs[i], cmap='bone')
            if grid_size < 13:
                ax.set_xlabel('negative', labelpad=0.1)
            i += 1
        else:
            plt.imshow(positive_imgs[j], cmap='bone')
            if grid_size < 13:
                ax.set_xlabel('positive',labelpad=0.1)
            j += 1
    plt.show()
    #plt.savefig(os.path.abspath(os.path.join('/content/drive/MyDrive/ColabNotebooks/BI-BAP', 'diagrams/covidx8_sample.pdf')), bbox_inches='tight', format='pdf')
    

def plot_eval(y_true, y_pred_prob, y_pred):
    """
    Prints a confusion matrix and an ROC curve with a calculated AUC metric for given classification results.
    
    Parameters
    ----------
    y_true : NumPy.ndarray
        Ground truth labels of the evaluated samples.
    y_pred_prob : NumPy.ndarray
        Predicted probabilities of the evaluated samples belonging to the positive class.
    y_pred : NumPy.ndarray
        Predicted class of the evaluated samples.
    """
    
    # set font style to match with written part of the thesis
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # construct a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'], vmin=0, vmax=np.count_nonzero(y_true == 0), annot_kws={'size': 15})

    ax.set_yticklabels(labels=ax.get_yticklabels(), va='center')
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('Predicted labels', fontsize = 14, labelpad=12)
    plt.ylabel('True labels', fontsize = 14, labelpad=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.show()

    # calculate ROC rurve and its AUC metric
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, color='darkblue')
    plt.plot(fpr_keras, tpr_keras, label='ROC (AUC = {:.3f})'.format(auc), linewidth=2, color='darkorange')

    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('False positive rate', fontsize=14, labelpad=12)
    plt.ylabel('True positive rate', fontsize=14, labelpad=8)
    plt.title('Receiver Operating Characteristic Curve', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.xlim(-0.01,1)
    plt.ylim(0,1.01)

    plt.show()

    
    
def get_generators(datagen, train_dir, test_dir, img_size=224, batch_size=32, channel_cnt=3, shuffle=True, rand_seed=111):
    """
    Creates data sources from a given image data generator, which are then used in training and evaluating a model.
    
    Parameters
    ----------
    datagen : TensorFlow.keras.preprocessing.image.ImageDataGenerator
        Image data generator based on which the sources will be created.
    train_dir : str
        Path to folder containing training images.
    test_dir : str
        Path to folder containing testing images.
    img_size : int
        Target size of the images. Default value: 224.
    batch_size : int
        Batch size, which will be yielded by the generators in each call. Default value: 32.
    channel_cnt : int
        Number of channels in the input images. The count determines whether rgb or grayscale mode will be used. Default value: 3.
    shuffle : bool
        Wheteher to shuffle the order of the images when generating them. Default value: True.
    rand_seed : int
        Random seed for the shuffling. Default value: 111.
    
    Returns
    -------
    train_gen, valid_gen, test_gen : TensorFlow.keras.preprocessing.image.DirectoryIterator
        Iterators, which yield tuples of (x, y) where x is a NumPy array containing a batch of images and y is a NumPy array of their corresponding labels.
    """
    if channel_cnt == 3:
        mode = 'rgb'
    elif channel_cnt == 1:
        mode = 'grayscale'
    else:
        raise ValueError(f'The target number of channels in the images must be either 1 (grayscale) or 3 (rgb)')
        
    train_gen = datagen.flow_from_directory(
                    directory=train_dir,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='binary',
                    subset='training',
                    color_mode=mode,
                    shuffle=shuffle,
                    seed=rand_seed
                )

    valid_gen = datagen.flow_from_directory(
                    directory=train_dir,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='binary',
                    subset='validation',
                    color_mode=mode,
                    shuffle=shuffle,
                    seed=rand_seed
                )
    
    # copy original datagen's preprocessing function, but discard augmentation settings
    datagen_test = ImageDataGenerator(preprocessing_function=datagen.preprocessing_function)
    test_gen = datagen_test.flow_from_directory(
                    directory=test_dir,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='binary',
                    color_mode=mode,
                    shuffle=False,
                    seed=rand_seed
                )
    
    return train_gen, valid_gen, test_gen


def get_class_weights(train_dir, class_cnt=2):
    """
    Calculates weights for the two classes based on their representation in the training data.
    The more observed class will be assigned a proportionally lower weight.
    
    Parameters
    ----------
    train_dir : str
        Path to the folder with training data. This folder should include two subfolders, one for each class (negative, positive).
    
    Returns
    -------
    class_weights : dict
        Dictionary where there is a key for both of the classes and the key's value is the weight for the given class in training.
        Negative class has key '0' and positive class key '1'.
    """
    
    negative_cnt = len(os.listdir(os.path.join(train_dir, 'negative')))
    positive_cnt = len(os.listdir(os.path.join(train_dir, 'positive')))
    total = negative_cnt + positive_cnt

    class_weights = {
        0: total / (negative_cnt * class_cnt),
        1: total / (positive_cnt * class_cnt)
    }
    
    return class_weights
    
    
def create_dataframe(train_dir):
    """
    Collects all files in a dataset folder into a DataFrame and labels them according to which subfolder (class) they belong to.
    
    Parameters
    ----------
    train_dir : str
        Path to folder which is to be turned into a DataFrame.
    
    Returns
    -------
    train_df : pandas.DataFrame
        Shuffled dataframe with all the files of given directory. Files are identified by the column 'path' and their label by the column 'label'.
    """
    train_df = pd.DataFrame(columns=['path', 'label'])
    labels = ['negative', 'positive']
    for l in labels:
        paths = os.listdir(os.path.join(train_dir, l))
        train_df = train_df.append([pd.Series({'path': os.path.join(l, i), 'label': l}) for i in paths], ignore_index=True)
    return train_df.sample(frac=1)
    

def get_crossval_generators(datagen, train_dir, train_data, valid_data, img_size=224, batch_size=32, channel_cnt=3, shuffle=True, rand_seed=111):
    """
    Creates data sources from a given image data generator, which are then used in training and validating a model. These sources use data indexed by two DataFrames
    and are meant to be applied in the K-fold cross-validation pipeline.
    
    Parameters
    ----------
    datagen : TensorFlow.keras.preprocessing.image.ImageDataGenerator
        Image data generator based on which the sources will be created.
    train_dir : str
        Path to folder containing training images.
    train_data : pandas.DataFrame
        Dataframe containing paths and labels of training data.
    valid_data : pandas.DataFrame
        Dataframe containing paths and labels of validation data.
    img_size : int
        Target size of the images. Default value: 224.
    batch_size : int
        Batch size, which will be yielded by the generators in each call. Default value: 32.
    channel_cnt : int
        Number of channels in the input images. The count determines whether rgb or grayscale mode will be used. Default value: 3.
    shuffle : bool
        Wheteher to shuffle the order of the images when generating them. Default value: True.
    rand_seed : int
        Random seed for the shuffling. Default value: 111.
    
    Returns
    -------
    train_gen, valid_gen : TensorFlow.keras.preprocessing.image.DirectoryIterator
        Iterators, which yield tuples of (x, y) where x is a NumPy array containing a batch of images and y is a NumPy array of their corresponding labels.
    """
    if channel_cnt == 3:
        mode = 'rgb'
    elif channel_cnt == 1:
        mode = 'grayscale'
    else:
        raise ValueError(f'The target number of channels in the images must be either 1 (grayscale) or 3 (rgb)')
        
    train_gen = datagen.flow_from_dataframe(
                    dataframe=train_data,
                    directory=train_dir,
                    x_col='path',
                    y_col='label',
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='binary',
                    color_mode=mode,
                    shuffle=shuffle,
                    seed=rand_seed
                )

    valid_gen = datagen.flow_from_dataframe(
                    dataframe=valid_data,
                    directory=train_dir,
                    x_col='path',
                    y_col='label',
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='binary',
                    color_mode=mode,
                    shuffle=shuffle,
                    seed=rand_seed
                )
    
    return train_gen, valid_gen
    

def oversampling(train_dir, img_size=224, batch_size=32, ratio=1, augment=True):
    """
    Oversample the minority class to perform class balancing in the training data. If augment is set to True, the re-added images will first be augmented.
    
    Parameters
    ----------
    train_dir : str
        Path to the training directory.
    img_size : int
        Target size of the images. Default value: 224.
    batch_size : int
        Batch size, which will be yielded by the generators in each call. Default value: 32.
    ratio : float
        Approximate ratio of the classes after the balancing is performed. Default value: 1.
    augment : bool
        Whether to perform augmentations on the oversampled images.
    """
    
    print(f"Before oversampling, the positive class has { len(os.listdir(os.path.join(train_dir, 'positive'))) } samples")
    
    source_dir = './sample_pool'
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
    shutil.copytree(os.path.join(train_dir, 'positive'), os.path.join(source_dir, 'positive'))

    if augment:
        datagen = ImageDataGenerator(
                #Choose data augmentation parameters
                rotation_range=10,
                width_shift_range=0.03,
                height_shift_range=0.05,
                horizontal_flip=False,
                brightness_range=(0.9, 1.1),
                zoom_range=(0.9, 1.1),
                fill_mode='constant',
                cval=0.
          )
    else:
        datagen = ImageDataGenerator()
    
    oversampler = datagen.flow_from_directory(
                    directory=source_dir,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='binary',
                    save_to_dir=os.path.join(train_dir, 'positive'),
                    save_prefix='new_',
                    color_mode='rgb',
                    shuffle=True,
                    seed=111
                    )

    negative_cnt = len(os.listdir(os.path.join(train_dir, 'negative')))
    while len(os.listdir(os.path.join(train_dir, 'positive'))) < (negative_cnt / ratio):
        oversampler.next()

    print(f"After oversampling the positive class, it now contains {len(os.listdir(os.path.join(train_dir, 'positive')))} samples")


@tf.function
def f1_metric(y_true, y_pred):
    """
    Function to calculate the F1 metric during continuous training of a TensorFlow model.
    Original source: # https://datascience.stackexchange.com/a/48251
    
    Parameters
    ----------
    y_true : NumPy.ndarray
        Ground truth labels of the evaluated batch.
    y_pred : NumPy.ndarray
        Predicted class of the evaluated batch.
        
    Returns
    -------
    f1_val : float
        F1 score for the positive class.
    """
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val