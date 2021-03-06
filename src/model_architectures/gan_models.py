"""
This file contains the definitions of functions that build and control the training 
of Deep Convolutional Generative Adversarial Networks (DCGANs).

Author: Dominik Chodounský
Institution: Faculty of Information Technology, Czech Technical University in Prague
Last edit: 2021-04-19
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Dense, Input, Layer, Flatten, Activation, GlobalAveragePooling2D, Dropout, ZeroPadding2D, Conv2D, MaxPool2D, Reshape, Conv2DTranspose, LeakyReLU, UpSampling2D, BatchNormalization, ReLU
from tensorflow.keras.optimizers import SGD, Nadam, Adam
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import time


def build_discriminator(img_size=128, channel_cnt=3):
    """
    Builds the discriminator of the DCGAN, which classifies the images as either synthetically generated or real.
    
    Parameters
    ----------
    img_size : int
        Size of the images that will be classified by the discriminator. Default value: 128.
    channel_cnt : int
        Number of channels of the images (3 for RGB, 1 for grayscale). Default value: 3.
        
    Returns
    -------
    model : TensorFlow.keras.Sequential()
        Compiled discriminator model.
    """
        
    model = Sequential(name='Discriminator')

    model.add(Conv2D(img_size, (5,5), padding='same', input_shape=(img_size, img_size, channel_cnt)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(img_size, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(img_size, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(img_size, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(img_size, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', name='D'))
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(lr=0.0002, beta_1=0.5), 
        metrics=['accuracy'])

    return model
    

def build_generator(img_size=128, channel_cnt=3):
    """
    Builds the generator of the DCGAN, which learns to generate synthetic images from a noise distribution.
    
    Parameters
    ----------
    img_size : int
        Size of the images that will be generated by the generator. Default value: 128.
    channel_cnt : int
        Number of channels of the images (3 for RGB, 1 for grayscale). Default value: 3.
        
    Returns
    -------
    model : TensorFlow.keras.Sequential()
        Compiled generator model.
    """
    
    model = Sequential(name='Generator')

    model.add(Dense(img_size * 16 * 16, use_bias=False, input_dim=(100)))
    model.add(ReLU())

    model.add(Reshape((16, 16, img_size)))
    assert model.output_shape == (None, 16, 16, img_size)
    
    model.add(Conv2DTranspose(img_size, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, img_size)
    model.add(ReLU())
    
    model.add(Conv2DTranspose(img_size//2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, img_size//2)
    model.add(ReLU())
    
    model.add(Conv2DTranspose(img_size//4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, img_size//4)
    model.add(ReLU())
    
    # perform one more upsampling if images are 256
    if img_size==256:
        model.add(Conv2DTranspose(img_size//8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 256, 256, img_size//8)
        model.add(ReLU())
    
    model.add(Conv2D(channel_cnt, (5, 5), activation='tanh', padding='same', use_bias=False, name='G'))
    assert model.output_shape == (None, img_size, img_size, channel_cnt)
    
    return model


def build_gan(gen_model, discr_model):
    """
    Builds a DCGAN model which combines the generator G and the discriminator D.
    
    Parameters
    ----------
    gen_model : TensorFlow.keras.Sequential()
        Generator part of the GAN.
    discr_model : TensorFlow.keras.Sequential()
        Discriminator part of the GAN.
        
    Returns
    -------
    model : TensorFlow.keras.Sequential()
        Compiled DCGAN model.
    """
    
    discr_model.trainable = False

    model = Sequential(name='DCGAN')
    model.add(gen_model)
    model.add(discr_model)

    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    return model


def generate_latent_vector(sample_cnt):
    """
    Generates a random noise vector from the latent space which then serves as input for the generator.
    A 100-dimensional hypersphere where each feature is drawn from the Gaussian distribution is used.
    
    Parameters
    ----------
    sample_cnt : int
        Number of samples that the generator will be generating.
    
    Returns
    -------
    x_input : NumPy.ndarray
        Array of floating-point samples from the standard normal distribution.
    """
    
    x_input = np.random.randn(100 * sample_cnt)
    x_input = x_input.reshape(sample_cnt, 100)
    return x_input


def generate_fake_samples(gen_model, sample_cnt):
    """
    Feeds sample from a noise distribution into the generator, which generates synthetic samples
    and labels them with the fake class label 0.
    
    Parameters
    ----------
    gen_model : TensorFlow.keras.Sequential()
        Generator model.
    sample_cnt : int
        Number of samples that the generator will be generating.
    
    Returns
    -------
    X : NumPy.ndarray
        Array of synthetic images generated by the generator.
    y : NumPy.ndarray
        Array of 0s which marks the fake class labels for all the images.
    """
    
    x_input = generate_latent_vector(sample_cnt)
    X = gen_model.predict(x_input)

    # add 'fake' class labels (0)
    y = np.zeros((sample_cnt, 1))
    return X, y


def read_images(train_dir, paths, img_size=128, channel_cnt=3):
    """
    Reads and preprocesses a batch of images to be used in training of the GAN.
    
    Parameters
    ----------
    train_dir : str
        Path to the training data directory, which contains real images.
    paths : NumPy.ndarray
        List of strings that contain the filenames of the images to be read into the current batch.
    img_size : int
        Size of the images that will be generated by the generator. Default value: 128.
    channel_cnt : int
        Number of channels of the images (3 for RGB, 1 for grayscale). Default value: 3.
    
    Returns
    -------
    imgs : NumPy.ndarray
        Array of images to be used in the current batch.
    """
    
    imgs = []
    for path in paths:
        img = cv2.imread(os.path.join(train_dir, path))

        if channel_cnt == 1 and len(img.squeeze().shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img.astype('uint8'), (img_size, img_size))
        img = np.asarray(img / 255)
        img = img.reshape((img_size, img_size, channel_cnt))
        imgs.append(img)
    return np.asarray(imgs)


def get_real_samples(train_dir, data, sample_cnt, img_size=128):
    """
    Produces a random batch of images from the training data and labels them as real images with class label 1.
    
    Parameters
    ----------
    train_dir : str
        Path to the training data directory, which contains real images.
    data : NumPy.ndarray
        An array of image filenames of the training data.
    sample_cnt : int
        Number of real image samples to be selected.
    img_size : int
        Size that the selected images will be resized to. Default value: 128.
    
    Returns
    -------
    X : NumPy.ndarray
        Array of real images sampled from the training data.
    y : NumPy.ndarray
        Array of 1s which marks the real class labels for all the image
    """
    
    ix = np.random.randint(0, data.shape[0], sample_cnt)
    X = read_images(train_dir, data[ix], img_size=img_size)

    # add 'real' class labels (1)
    y = np.ones((sample_cnt, 1))
    return X, y


def show_generated(generated, n=5, channel_cnt=3):
    """
    Plot a grid of images to display as an example.
    
    Parameters
    ----------
    generated : NumPy.ndarray
        Array of images that are to be plotted (function is meant to plot images generated by the generator).
    n : int
        Size of the grid. Default value: 5.
    channel_cnt : int
        Number of channels of the images. Default value: 3.
    """
    
    plt.figure(figsize=(10,10))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        if channel_cnt == 1:
            plt.imshow(np.clip(generated[i], 0, 1).squeeze(), vmin=0, vmax=1, cmap='gray')
        else:
            plt.imshow(np.clip(generated[i], 0, 1), vmin=0, vmax=1)
        plt.axis('off')
    plt.show()    


def summary(gen_model, discr_model, train_dir, data, img_size=128, sample_cnt=100):
    """
    Function shows a summary of the progress of the training, including the discriminator's accuracy on the real images
    and on the synthetic images. It also displays examples of the generated images.
    
    gen_model : TensorFlow.keras.Sequential()
        Generator model.
    discr_model : TensorFlow.keras.Sequential()
        Discriminator model.
    train_dir : str
        Path to the training data directory, which contains real images.
    data : NumPy.ndarray
        Array containing filenames of the training data images.
    img_size : int
        Size of the images used in training. Default value: 128.
    sample_cnt : int
        Number of synthetic images to generate. Default value: 100.
    """
    
    X_real, y_real = get_real_samples(train_dir, data, sample_cnt, img_size)

    _, acc_real = discr_model.evaluate(X_real, y_real, verbose=0)

    x_fake, y_fake = generate_fake_samples(gen_model, sample_cnt)

    _, acc_fake = discr_model.evaluate(x_fake, y_fake, verbose=0)

    print(f'Accuracy => Real: {acc_real:.2%},   Fake: {acc_fake:.2%}')
    show_generated(x_fake)


def continue_summary(gen_model, discr_model, train_dir, data, img_size=128, sample_cnt=100):
    """
    Function shows a summary of the progress of continued training, including the discriminator's accuracy on the real images
    and on the synthetic images. It also displays examples of the generated images.
    
    Note: Function is a helper function which only slightly changes the original summary function so that it is able to be used
    when continuing to train a saved model.
    
    gen_model : TensorFlow.keras.Sequential()
        Generator model.
    discr_model : TensorFlow.keras.Sequential()
        Discriminator model.
    train_dir : str
        Path to the training data directory, which contains real images.
    data : NumPy.ndarray
        Array containing filenames of the training data images.
    img_size : int
        Size of the images used in training. Default value: 128.
    sample_cnt : int
        Number of synthetic images to generate. Default value: 100.
    """

    X_real, y_real = get_real_samples(train_dir, data, sample_cnt, img_size)

    acc_real = discr_model.evaluate(X_real, y_real, verbose=0)

    x_fake, y_fake = generate_fake_samples(gen_model, sample_cnt)

    acc_fake = discr_model.evaluate(x_fake, y_fake, verbose=0)

    print(f'Accuracy => Real: {acc_real:.2%},   Fake: {acc_fake:.2%}')
    show_generated(x_fake)  
    

def train(gen_model, discr_model, gan_model, train_dir, data, img_size=128, epoch_start=0, epoch_cnt=100, batch_cnt=16, save=''):
    """
    Training loop for the GAN. For each epoch it samples from the training data, generates images based on generated 
    inputs from the noise distribution and trains the discriminator on this batch. The model is then evaluated after every 5th epoch.
    
    gen_model : 
        Generator model.
    discr_model : 
        Discriminator model.
    gan_model : 
        GAN model.
    train_dir : str
        Path to the training data directory, which contains real images.
    data : NumPy.ndarray
        Array of filenames of the images in the training data.
    img_size : int
        Size of the images used in training. Default value: 128.
    epoch_start : int
        The epoch that the training will start from. Default value: 0.
    epoch_cnt : int
        Total number of epochs that the model will be trained for. Default value: 100.
    batch_cnt : int
        Total size of the batch of images (includes synthetic and real). Default value: 16.
    save : str
        Path and filename where the model will be saved upon every evaluation.
    """
    
    bat_per_epo = int(data.shape[0] / batch_cnt)
    
    # split batch into two halves (one for synthetic, one for real)
    half_batch = int(batch_cnt / 2)

    start = time.time()
    
    # training from the beginning
    if epoch_start == 0:
        for i in range(epoch_cnt):
            for j in range(bat_per_epo):
        
                X_real, y_real = get_real_samples(train_dir, data, half_batch, img_size)
    
                d_loss1, _ = discr_model.train_on_batch(X_real, y_real)
     
                X_fake, y_fake = generate_fake_samples(gen_model, half_batch)
    
                d_loss2, _ = discr_model.train_on_batch(X_fake, y_fake)
       
                X_gan = generate_latent_vector(batch_cnt)
        
                y_gan = np.ones((batch_cnt, 1))
        
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                print('.', end='')
       
            print(f'\nEpoch {i + 1}: Loss: D_real_loss = {d_loss1:.3}, D_fake_loss = {d_loss2:.3},  G_loss = {g_loss:.3}')
     
            if (i + 1) % 5 == 0:
                tf.keras.models.save_model(gan_model, save + f'-epoch_{i+1}.h5')
                summary(gen_model, discr_model, train_dir, data, img_size)
    
    # continue training a pre-loaded model
    else:
        for i in range(epoch_start, epoch_cnt):
            for j in range(bat_per_epo):
        
                X_real, y_real = get_real_samples(train_dir, data, half_batch, img_size)
    
                d_loss1 = discr_model.train_on_batch(X_real, y_real)
     
                X_fake, y_fake = generate_fake_samples(gen_model, half_batch)
    
                d_loss2 = discr_model.train_on_batch(X_fake, y_fake)
       
                X_gan = generate_latent_vector(batch_cnt)
        
                y_gan = np.ones((batch_cnt, 1))
        
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                print('.', end='')
       
            print(f'\nEpoch {i + 1}: Loss: D_real = {d_loss1:.3}, D_fake = {d_loss2:.3},  G = {g_loss:.3}')
     
            if (i + 1) % 5 == 0:
                tf.keras.models.save_model(gan_model, save + f'-epoch_{i+1}.h5')
                continue_summary(gen_model, discr_model, train_dir, data, img_size)
                
    print(f'Total training time for {epoch_cnt} epochs: {time.time()-start}')
    


