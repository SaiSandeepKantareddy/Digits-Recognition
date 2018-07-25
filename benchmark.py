"""
Created on: Fri May 31 12:30:23 2018
Author:     Jagan Seshadri
Summary:    Used for calculating the mAP on test dataset.

Usage: 
python benchmark.py \
--path ../dataset/test \
--model models/model.chkpt.hdf5
"""

# import
import os, argparse
import cv2
import numpy as np

from utils import image_generator
from tensorflow.python.keras.models import load_model
import pandas as pd

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to test dir")
ap.add_argument("-m", "--model", required=True,
	help="path to model")
args = ap.parse_args()

mAP = []

for _p in os.listdir(args.path):
    _p = os.path.join(args.path, _p)
    #perform the prediction
    model = load_model(args.model)
    steps = len(os.listdir(_p))
    label = int(os.path.split(_p)[1])
    predictions = model.predict_generator(image_generator(_p), steps = steps)
    error = np.mean(np.argmax(predictions, axis=-1) != label)
    mAP.append((label, error))

with open('results.txt', 'a') as f:
    for label, error in mAP:
        mAP_str = "Class: {}, Error Rate: {}\n".format(label,error)
        print(mAP_str)
        f.write(mAP_str)
    mAP = 1 - (sum(err for lb, err in mAP) / len(mAP))
    f.write("mAP: {:.4f}\n".format(mAP*100))

print("mAP: ", mAP)