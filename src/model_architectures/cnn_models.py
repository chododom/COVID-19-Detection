"""
This file contains the definitions of functions that build and control the training 
of Convolutional Neural Networks.

Author: Dominik Chodounský
Institution: Faculty of Information Technology, Czech Technical University in Prague
Last edit: 2021-04-07
"""


import os
import tensorflow as tf
from tensorflow.keras import layers, Model, metrics, losses
from tensorflow.keras.layers import Dense, Input, Layer, Flatten, Activation, GlobalAveragePooling2D, Dropout, ZeroPadding2D, Conv2D, DepthwiseConv2D, concatenate, MaxPool2D, Reshape, Conv2DTranspose, LeakyReLU, UpSampling2D, BatchNormalization, add
from tensorflow.keras.optimizers import SGD, Nadam, Adam, RMSprop
from keras.applications import VGG16, VGG19, ResNet50, Xception, MobileNetV2, DenseNet121


class CNN:
    """
    A class used to represent a general Convolutional Neural Network.

    ...

    Attributes
    ----------
    name : str
        Name of the CNN.
    img_size : int
        Size of the images that the network is initialized to accept. Default value: 224.
    channel_cnt : int
        Number of channels of the images that the network is initialized to accept. Default value: 3.
    feature_extractor : TensorFlow.keras.layers.Layer
        The layers belonging to the convolutional base which performs feature extraction.
    weights : str
        Description of the pre-loaded weights (random, imagenet, chexnet). Default value: random.
  
    Methods
    -------
    summary()
        Prints the summary of the CNN.
        
    lock(lock_until=None)
        Allows for locking layers in the network so that they are not trainable.
        By leaving lock_until=None, the whole convolutional base (feature_extractor) is locked.
        If lock_until is specified, all layers from the first until the specified number are locked.
        
    unlock(unlock_from=None)
        Allows for unlocking layers in the network so that they are trainable if they have been locked before.
        By leaving unlock_from=None, the whole network is unlocked.
        If unlock_from is specified, all layers from the specified layer until the last are unlocked.
        
    compile(optimizer=Adam(learning_rate=0.0001), metrics=[metrics.BinaryAccuracy()])
        Compiles the CNN model.
    """
    
    def __init__(self, name, img_size=224, channel_cnt=3, weights='random'):
        """
        Parameters
        ----------
        name : str
            Name of the CNN.
        img_size : int
            Size of the images that the network will accept. Default value: 224.
        channel_cnt : int
            Number of channels of the images that the network will accept. Default value: 3.
        weights : str
            Weights to initialize the network with. Default value: random.
        """
        
        self.name = name
        self.img_size = img_size
        self.channel_cnt = channel_cnt
        self.feature_extractor = None
        self.weights = weights
    
    def summary(self):
        """
        Prints the summary of the CNN.
        """
        
        self.model.summary()
        
    def lock(self, lock_until=None):
        """
        Allows for locking layers in the network so that they are not trainable.
        By leaving lock_until=None, the whole convolutional base (feature_extractor) is locked.
        If lock_until is specified, all layers from the first until the specified number are locked.
        
        Params
        ------
        lock_until : int
            Specifies the last layer that will be locked. Default value: None.
        """
        
        if self.name == 'basenet' or self.name == 'covidnet':
            print('Locking not supported for this architecture!')
            return None
        if lock_until == None:
            self.feature_extractor.trainable = False
    
        self.feature_extractor.trainable = True
        for layer in self.feature_extractor.layers[:lock_until]:
            layer.trainable = False       
        
    def unlock(self, unlock_from=None):
        """
        Allows for unlocking layers in the network so that they are trainable if they have been locked before.
        By leaving unlock_from=None, the whole network is unlocked.
        If unlock_from is specified, all layers from the specified layer until the last are unlocked.
        
        Params
        ------
        unlock_from : int
            Specifies the first layer that will be unlocked. Default value: None.
        """
        
        if self.name == 'basenet' or self.name == 'covidnet':
            print('Unlocking not supported for this architecture!')
            return None
        if unlock_from == None:
            self.feature_extractor.trainable = True
        
        self.feature_extractor.trainable = False
        for layer in self.feature_extractor.layers[unlock_from:]:
            layer.trainable = True  

    def compile(self, optimizer=Adam(learning_rate=0.0001), metrics=[metrics.BinaryAccuracy()]):
        """
        Compiles the CNN model.
        
        Params
        ------
        optimizer : TensorFlow.keras.optimizers.Optimizer
            Optimizer to compile the model with. Default value: Adam(learning_rate=0.0001).
        metrics : list
            List of TensorFlow metrics to include in monitoring during training. Default value: [TensorFlow.keras.metrics.BinaryAccuracy()]
        """
        
        self.model.compile(
            optimizer=optimizer,
            metrics=metrics, 
            loss=losses.BinaryCrossentropy() 
        )


class BaseNet(CNN):
    """
    A class used to represent an implementation of our prototype BaseNet architecture for binary detection.
    Only allows random weight initialization.
    Architecture was optimized with Hyperband hyperparameter optimization.
    Does not allow locking/unlocking.
    """
    
    def __init__(self, img_size, channel_cnt, weights='random'):
        CNN.__init__(self, 'basenet', img_size, channel_cnt, weights)
        
        if self.weights != 'random':
            raise ValueError(f'BaseNet does not support pretrained weights, random weight initialization must be used.')
        
        dropout_hp = 0.2
        activation_hp = tf.keras.activations.relu
        
        input_layer = Input(shape=(self.img_size, self.img_size, self.channel_cnt), name='input')
                
        x1 = Conv2D(filters=64, kernel_size=5, strides=1, activation=activation_hp, padding="same")(input_layer)
        x1 = Conv2D(filters=64, kernel_size=5, strides=1, activation=activation_hp, padding="same")(x1)
        x1 = MaxPool2D(2, strides=2)(x1)
        x1 = Dropout(dropout_hp)(x1)
        
        x2 = Conv2D(filters=128, kernel_size=3, strides=1, activation=activation_hp, padding="same")(x1)
        x2 = Conv2D(filters=128, kernel_size=3, strides=1, activation=activation_hp, padding="same")(x2)
        x2 = Conv2D(filters=128, kernel_size=3, strides=1, activation=activation_hp, padding="same")(x2)
        x2 = MaxPool2D(2, strides=2)(x2)
        x2 = Dropout(dropout_hp)(x2)
        
        residual = Conv2D(filters=128, kernel_size=5, strides=1, activation=activation_hp, padding="same")(x2)
        residual = MaxPool2D(2, strides=8)(residual)
        
        
        x3 = Conv2D(filters=256, kernel_size=5, strides=1, activation=activation_hp, padding="same")(x2)
        x3 = Conv2D(filters=256, kernel_size=5, strides=1, activation=activation_hp, padding="same")(x3)
        x3 = MaxPool2D(2, strides=2)(x3)
        x3 = Dropout(dropout_hp)(x3)
        
        x4 = Conv2D(filters=256, kernel_size=5, strides=1, activation=activation_hp, padding="same")(x3)
        x4 = Conv2D(filters=256, kernel_size=5, strides=1, activation=activation_hp, padding="same")(x4)
        x4 = MaxPool2D(2, strides=2)(x4)
        x4 = Dropout(dropout_hp)(x4)
        
        x5 = Conv2D(filters=128, kernel_size=3, strides=1, activation=activation_hp, padding="same")(x4)
        x5 = Conv2D(filters=128, kernel_size=3, strides=1, activation=activation_hp, padding="same")(x5)
        x5 = Conv2D(filters=128, kernel_size=3, strides=1, activation=activation_hp, padding="same")(x5)
        x5 = MaxPool2D(2, strides=2)(x5)
        x5 = Dropout(dropout_hp)(x5)
        
        x6 = Conv2D(filters=128, kernel_size=5, strides=1, activation=activation_hp, padding="same", name='residual')(add([residual, x5]))
        x6 = Conv2D(filters=128, kernel_size=5, strides=1, activation=activation_hp, padding="same")(x6)
        x6 = Conv2D(filters=128, kernel_size=5, strides=1, activation=activation_hp, padding="same")(x6)
        x6 = Dropout(dropout_hp)(x6)
        
        x = GlobalAveragePooling2D(name='GAP')(x6)
        x = Flatten()(x)
        
        x = Dense(512, activation=activation_hp)(x)
        x = Dropout(dropout_hp)(x)

        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=[input_layer], outputs=[output], name='BaseNet')
        

class VGG_16(CNN):
    """
    A class used to create a VGG16 model for binary detection. The convolutional base is made from a pre-loaded VGG16,
    which is followed by our classifier.
    Allows random weight initialization and weights pre-trained on ImageNet.
    Allows locking/unlocking the convolutional base.
    """
    
    def __init__(self, img_size, channel_cnt, weights='random'):
        CNN.__init__(self, 'vgg_16', img_size, channel_cnt, weights)
        
        if self.channel_cnt != 3 and self.weights != 'random':
            raise ValueError(f'VGG_16 was pretrained on 3 channels, you selected {self.channel_cnt}. Pretrained weights cannot be used.')
        
        if self.channel_cnt != 3 or self.weights == 'random':
            vgg_16 = VGG16(include_top=False, weights=None, pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        elif self.weights == 'imagenet':    
            vgg_16 = VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        else:
            raise ValueError('Invalid model configuration.')

        self.feature_extractor = vgg_16
        x = Flatten()(vgg_16.layers[-1].output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)

        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=[vgg_16.inputs], outputs=[output], name='VGG16')
        

class VGG_19(CNN):
    """
    A class used to create a VGG19 model for binary detection. The convolutional base is made from a pre-loaded VGG19,
    which is followed by our classifier.
    Allows random weight initialization and weights pre-trained on ImageNet.
    Allows locking/unlocking the convolutional base.
    """
    
    def __init__(self, img_size, channel_cnt, weights='random'):
        CNN.__init__(self, 'vgg_19', img_size, channel_cnt, weights)
        
        if self.channel_cnt != 3 and self.weights != 'random':
            raise ValueError(f'VGG_19 was pretrained on 3 channels, you selected {self.channel_cnt}. Pretrained weights cannot be used.')
        
        if self.channel_cnt != 3 or self.weights == 'random':
            vgg_19 = VGG19(include_top=False, weights=None, pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        elif self.weights == 'imagenet':  
            vgg_19 = VGG19(include_top=False, weights='imagenet', pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        else:
            raise ValueError('Invalid model configuration.')

        self.feature_extractor = vgg_19
        x = Flatten()(vgg_19.layers[-1].output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)

        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=[vgg_19.inputs], outputs=[output], name='VGG19')



class ResNet_50(CNN):
    """
    A class used to create a ResNet50 model for binary detection. The convolutional base is made from a pre-loaded ResNet50,
    which is followed by our classifier.
    Allows random weight initialization and weights pre-trained on ImageNet.
    Allows locking/unlocking the convolutional base.
    """
    
    def __init__(self, img_size, channel_cnt, weights='random'):
        CNN.__init__(self, 'resnet_50', img_size, channel_cnt, weights)
        
        if self.channel_cnt != 3 and self.weights != 'random':
            raise ValueError(f'ResNet_50 was pretrained on 3 channels, you selected {self.channel_cnt}. Pretrained weights cannot be used.')
        
        if self.channel_cnt != 3 or self.weights == 'random':
            resnet_50 = ResNet50(include_top=False, weights=None, pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        elif self.weights == 'imagenet':  
            resnet_50 = ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        else:
            raise ValueError('Invalid model configuration.')

        self.feature_extractor = resnet_50
        x = Flatten()(resnet_50.layers[-1].output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)

        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=[resnet_50.inputs], outputs=[output], name='ResNet-50')
        

class xCeption(CNN):
    """
    A class used to create an Xception model for binary detection. The convolutional base is made from a pre-loaded Xception,
    which is followed by our classifier.
    Allows random weight initialization and weights pre-trained on ImageNet.
    Allows locking/unlocking the convolutional base.
    """
    
    def __init__(self, img_size, channel_cnt, weights='random'):
        CNN.__init__(self, 'xception', img_size, channel_cnt, weights)
        
        if self.channel_cnt != 3 and self.weights != 'random':
            raise ValueError(f'xCeption was pretrained on 3 channels, you selected {self.channel_cnt}. Pretrained weights cannot be used.')
        
        if self.channel_cnt != 3 or self.weights == 'random':
            xception = Xception(include_top=False, weights=None, pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        elif self.weights == 'imagenet':    
            xception = Xception(include_top=False, weights='imagenet', pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        else:
            raise ValueError('Invalid model configuration.')
            
        self.feature_extractor = xception
        x = Flatten()(xception.layers[-1].output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)

        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=[xception.inputs], outputs=[output], name='Xception')


class MobileNet_V2(CNN):
    """
    A class used to create a MobileNetV2 model for binary detection. The convolutional base is made from a pre-loaded MobileNetV2,
    which is followed by our classifier.
    Allows random weight initialization and weights pre-trained on ImageNet.
    Allows locking/unlocking the convolutional base.
    
    [This model was not used in our reported experimentation]
    """
    
    def __init__(self, img_size, channel_cnt, alpha=1.0, weights='random'):
        CNN.__init__(self, 'mobilenet_v2', img_size, channel_cnt, weights)
        
        if self.channel_cnt != 3 and self.weights != 'random':
            raise ValueError(f'MobileNet_V2 was pretrained on 3 channels, you selected {self.channel_cnt}. Pretrained weights cannot be used.')
        
        if self.channel_cnt != 3 or self.weights == 'random':
            mobilenet_v2 = MobileNetV2(include_top=False, alpha=alpha, weights=None, pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        elif self.weights == 'imagenet':    
            mobilenet_v2 = MobileNetV2(include_top=False, alpha=alpha, weights='imagenet', pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
        else:
            raise ValueError('Invalid model configuration.')
        
        self.feature_extractor = mobilenet_v2
        x = Flatten()(mobilenet_v2.layers[-1].output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)

        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=[mobilenet_v2.inputs], outputs=[output], name='MobileNet-V2')
        

class DenseNet_121(CNN):
    """
    A class used to create an DenseNet121 model for binary detection. The convolutional base is made from a pre-loaded DenseNet121,
    which is followed by our classifier.
    Allows random weight initialization and weights pre-trained on ImageNet and ChestX-ray14 (model known as CheXNet).
    Allows locking/unlocking the convolutional base.
    """
    
    def __init__(self, img_size, channel_cnt, weights='random', weights_path=''):
        CNN.__init__(self, 'densenet_121', img_size, channel_cnt, weights)
        
        if self.channel_cnt != 3 and self.weights != 'random':
            raise ValueError(f'DenseNet_121 was pretrained on 3 channels, you selected {self.channel_cnt}. Pretrained weights cannot be used.')
        
        if self.channel_cnt != 3 or self.weights == 'random':
            densenet_121 = DenseNet121(include_top=False, weights=None, pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
            self.feature_extractor = densenet_121
            x = Flatten()(densenet_121.layers[-1].output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.2)(x)
            
            output = Dense(1, activation='sigmoid')(x)
            
            self.model = Model(inputs=[densenet_121.inputs], outputs=[output], name='DenseNet-121')
        
        elif self.weights == 'chexnet':
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f'Path to CheXNet weights {weights_path} is invalid.')
            
            densenet_121 = DenseNet121(include_top=False, weights=None, pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
            output = tf.keras.layers.Dense(14, activation='sigmoid', name='output')(densenet_121.layers[-1].output)
            chexnet = tf.keras.Model(inputs=[densenet_121.input], outputs=[output])
            chexnet.load_weights(weights_path)
            
            self.feature_extractor = chexnet
            x = Flatten()(chexnet.layers[-2].output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.2)(x)
            
            output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
            
            self.model = tf.keras.Model(inputs=[chexnet.input], outputs=[output], name='DenseNet-121')
            
        elif self.weights == 'imagenet':    
            densenet_121 = DenseNet121(include_top=False, weights='imagenet', pooling='max', input_shape=(self.img_size, self.img_size, self.channel_cnt))
            self.feature_extractor = densenet_121
            x = Flatten()(densenet_121.layers[-1].output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.2)(x)
            
            output = Dense(1, activation='sigmoid')(x)
            
            self.model = Model(inputs=[densenet_121.inputs], outputs=[output], name='DenseNet-121')
        else:
            raise ValueError('Invalid model configuration.')


def PEPX_module(input_tensor, filters, strides=(1,1), name='PEPX'):
    """ 
    PEPX block as described by the COVID-Net publication: https://arxiv.org/abs/2003.09871
    
    Structure of the block:
    1. First-stage Projection: 1x1 convolutions for projecting input features to a lower dimension.
    2. Expansion: 1x1 convolutions for expanding features to a higher dimension that is different than that of the input features
    3. Depth-wise Representation: efficient 3x3 depth-wise convolutions for learning spatial characteristics to minimize computational complexity while preserving representational capacity.
    4. Second-stage Projection: 1x1 convolutions for projecting features back to a lower dimension.
    5. Extension: 1x1 convolutions that finally extend channel dimensionality to a higher dimension to produce the final features.
    """
    
    x = Conv2D(filters=filters, kernel_size=(1,1), activation='relu', strides=strides, name=name + '_projection_1')(input_tensor)   
    x = Conv2D(filters=filters, kernel_size=(1,1), activation='relu', name=name + '_expansion')(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same', name=name + '_depth_wise_representation')(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + '_projection_2')(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + '_extensions')(x)
    return x
    

class COVIDNet(CNN):
    """
    A class used to create our implementation of the COVID-Net models described in https://arxiv.org/abs/2003.09871 and published at https://github.com/lindawangg/COVID-Net/blob/master/docs/models.md.
    In our codebase, we call the network 'covidnet', but in the accompanying written thesis, we call it the COVID-Net CXR3-B2.
    It imitates the PEPX block design pattern, but makes some adaptations to the original COVID-Net CXR3-B which downsize the input layer and make the architecture fit for binary detection.
    Only allows random weight initialization.
    Does not allow locking/unlocking the convolutional base.
    """
    
    def __init__(self, img_size, channel_cnt, weights='random'):
        CNN.__init__(self, 'covidnet', img_size, channel_cnt, weights)
        
        if self.weights != 'random':
            raise ValueError(f'COVIDNet does not support pretrained weights, random weight initialization must be used.')
        
        input_layer = Input(shape=(img_size, img_size, channel_cnt), name='input')
        x = Conv2D(filters=56, kernel_size=(7,7), activation='relu', padding='same', strides=(2,2), name='conv7x7')(input_layer)
                
        # STAGE 1
        pepx1_c = Conv2D(filters=56, kernel_size=(1,1), activation='relu', padding='same', strides=(2,2), name='PEPX1_conv')(x)
                
        pepx1_1 = PEPX_module(input_tensor=x, filters=56, strides=(2,2), name='PEPX1_1')
        pepx1_2 = PEPX_module(input_tensor=concatenate([pepx1_c, pepx1_1]), filters=56, name='PEPX1_2')
        pepx1_3 = PEPX_module(input_tensor=concatenate([pepx1_c, pepx1_1, pepx1_2]), filters=56, name='PEPX1_3')
                
        # STAGE 2
        pepx2_c = Conv2D(filters=112, kernel_size=(1,1), activation='relu', padding='same', strides=(2,2), name='PEPX2_conv')(concatenate([pepx1_c, pepx1_1, pepx1_2, pepx1_3]))
                
        pepx2_1 = PEPX_module(input_tensor=concatenate([pepx1_c, pepx1_1, pepx1_2, pepx1_3]), filters=112, strides=(2,2), name='PEPX2_1')
        pepx2_2 = PEPX_module(input_tensor=concatenate([pepx2_c, pepx2_1]), filters=112, name='PEPX2_2')
        pepx2_3 = PEPX_module(input_tensor=concatenate([pepx2_c, pepx2_1, pepx2_2]), filters=112, name='PEPX2_3')
        pepx2_4 = PEPX_module(input_tensor=concatenate([pepx2_c, pepx2_1, pepx2_2, pepx2_3]), filters=112, name='PEPX2_4')
                
        # STAGE 3
        pepx3_c = Conv2D(filters=224, kernel_size=(1,1), activation='relu', padding='same', strides=(2,2), name='PEPX3_conv')(concatenate([pepx2_c, pepx2_1, pepx2_2, pepx2_3, pepx2_4]))
                
        pepx3_1 = PEPX_module(input_tensor=concatenate([pepx2_c, pepx2_1, pepx2_2, pepx2_3, pepx2_4]), filters=216, strides=(2,2), name='PEPX3_1')
        pepx3_2 = PEPX_module(input_tensor=concatenate([pepx3_c, pepx3_1]), filters=224, name='PEPX3_2')
        pepx3_3 = PEPX_module(input_tensor=concatenate([pepx3_c, pepx3_1, pepx3_2]), filters=216, name='PEPX3_3')
        pepx3_4 = PEPX_module(input_tensor=concatenate([pepx3_c, pepx3_1, pepx3_2, pepx3_3]), filters=216, name='PEPX3_4')
        pepx3_5 = PEPX_module(input_tensor=concatenate([pepx3_c, pepx3_1, pepx3_2, pepx3_3, pepx3_4]), filters=216, name='PEPX3_5')
        pepx3_6 = PEPX_module(input_tensor=concatenate([pepx3_c, pepx3_1, pepx3_2, pepx3_3, pepx3_4, pepx3_5]), filters=224, name='PEPX3_6')
                
        # Stage 4
        pepx4_c = Conv2D(filters=424, kernel_size=(1,1), activation='relu', padding='same', strides=(2,2), name='PEPX4_conv')(concatenate([pepx3_c, pepx3_1, pepx3_2, pepx3_3, pepx3_4, pepx3_5, pepx3_6]))
                
        pepx4_1 = PEPX_module(input_tensor=concatenate([pepx3_c, pepx3_1, pepx3_2, pepx3_3, pepx3_4, pepx3_5, pepx3_6]), filters=424, strides=(2,2), name='PEPX4_1')
        pepx4_2 = PEPX_module(input_tensor=concatenate([pepx4_c, pepx4_1]), filters=424, name='PEPX4_2')
        pepx4_3 = PEPX_module(input_tensor=concatenate([pepx4_c, pepx4_1, pepx4_2]), filters=400, name='PEPX4_3')
        
        x = Flatten()(concatenate([pepx4_c, pepx4_1, pepx4_2, pepx4_3]))
        #x = Dense(3, activation='relu')(x)
                
        output = Dense(1, activation='sigmoid')(x)
                
        self.model = Model(inputs=[input_layer], outputs=[output], name='COVID-Net-CXR3-B2')
