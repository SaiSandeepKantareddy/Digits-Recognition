"""
Created on: Fri Jun 15 11:20:11 2018
Author:     Jagan Seshadri
Summary:    Converts the model from keras to tensorflow format.

Usage:
python convert_model.py \
--model models/model.chkpt.hdf5 \
--checkpoint models/converted_model/model.chkpt
"""
# imports
import argparse
import tensorflow as tf

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to model")
ap.add_argument("-c", "--checkpoint", required=True,
	help="path to save checkpoint")
args = ap.parse_args()

# load model
model = load_model(args.model)
saver= tf.train.Saver()
with K.get_session() as sess:
    # save the session.
    save_path = saver.save(sess, args.checkpoint)
    print("Model saved in path: %s" % save_path)