"""
Created on: Fri May 31 12:30:23 2018
Author:     Jagan Seshadri
Summary:    This script finetunes the model pre-trained on mnist to fit the 
            custom dataset.
Usage: 
python finetune.py
"""
# imports
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K

from utils import custom_data_generator

# training params
batch_size = 4
epochs = 10 
train_dir = '../dataset/train'
val_dir = '../dataset/val'
target_size = 28, 28
train_sample_count = 267
val_sample_count =  31

# load model
model = load_model('models/mnist_pretrained.hdf5')

# load data
train_gen = custom_data_generator(train_dir, target_size, 
                                  color_mode = 'grayscale',
                                  batch_size = 32,
                                  augment=True)

val_gen = custom_data_generator(val_dir, target_size, 
                                color_mode = 'grayscale',
                                batch_size = 1)

# finetuning
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

mcp = ModelCheckpoint("models/model.chkpt.hdf5", save_best_only=True)

# defining class weights for unbalanced training set.
class_weights = {0:1.612,   
                 1:2.631,
                 2:1.428,
                 3:1.667,
                 4:1.852,
                 5:1.428,
                 6:1.852,
                 7:2.273,
                 8:1.000,
                 9:2.273}

# training
history = model.fit_generator(train_gen,
                    steps_per_epoch = train_sample_count // batch_size,
                    epochs=50,
                    callbacks=[mcp],
                    validation_data = val_gen,
                    validation_steps = val_sample_count // batch_size,
                    class_weight = class_weights)
