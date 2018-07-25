"""
Created on: Fri May 31 12:30:23 2018
Author:     Jagan Seshadri
Summary:    Utility functions.
"""

# imports
from time import time
import argparse, os, functools

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# methods

def crop(img,n,crop_size=28):
    '''Yields `n` equidistant crops of `img`.
    '''
    for i in range(n):
        yield img[:,i*crop_size:(i+1)*crop_size]

def timit(function):
    '''Decorator used for timing functions.
    '''
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time()
        function(*args, **kwargs)
        end = time()
        print("Time taken to run `{}` is {:.2f}s.".format(function.__name__, end-start))
    return wrapper

def custom_data_generator(directory, target_size, 
                          color_mode = 'rgb', batch_size = 32, 
                          shuffle = False, seed = 0,
                          save_dir = None, sav_pfix='',
                          augment = False):
    '''
    Keras wrapper for 'flow_from_directory' method.
    '''
    datagen = ImageDataGenerator(rescale=1./255)

    if augment: 
        datagen = ImageDataGenerator(rescale=1./255,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     fill_mode='nearest')

    return datagen.flow_from_directory(directory,
                            target_size=target_size,
                            color_mode=color_mode,
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            seed=seed,
                            save_to_dir=save_dir,
                            save_prefix=sav_pfix)


def image_generator(path):
    '''
    Workaround image generator.
    '''
    for p in os.listdir(path):
        p = os.path.join(path, p)
	# read image as greyscale
        img = cv2.imread(p,0)
	# resize image
        img = cv2.resize(img, (28,28), cv2.INTER_AREA)
	# covert to 4-dim tensor
        img = img.reshape(1,28,28,1) 
	# perform normalization
        img = img.astype('float32')
        img /= 255.
        yield (img)