"""
Created on: Fri May 31 12:30:23 2018
Author:     Jagan Seshadri
Summary:    Main file, contains inference pipeline.

Usage: 
python main.py \
--image ../example_images/test.jpg \
--model models/model.chkpt.hdf5 \
--digits 4 \
--precision 2
"""
# imports
import cv2
import numpy as np
import argparse
from utils import crop, timit
import tensorflow as tf
from tensorflow.python.keras.models import load_model

@timit
def inference(image, model, n, p):
    '''
    Performs inference.
    '''
    # read the image
    x = cv2.imread(image, 0)
    print(x.shape)
    # resize
    x = cv2.resize(x,(28*n,28),cv2.INTER_AREA)
    # get the crops
    x = np.vstack([i[None] for i in crop(x,n,28)])
    # convert to a 4D tensor
    x = x.reshape(n,28,28,1)
    # normalize it
    x = x.astype('float32')
    x /= 255
    
    #perform the prediction    
    model = load_model(args.model)
    out = model.predict_on_batch(x)

    # Add the precision
    output = map(str, np.argmax(out,-1).tolist())
    output = ''.join(output)
    if p == 0: output =  output + '.'
    elif p == n: output = '.' + output
    else: output = output[:-p] + '.' + output[-p:]
    print(output)

if __name__ =="__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
    	help="path to input image")
    ap.add_argument("-m", "--model", required=True,
    	help="path to model")
    ap.add_argument("-n", "--digits", required=True,
    	help="number of digits")
    ap.add_argument("-p", "--precision", required=True,
    	help="position of decimal, ex: 1 = 0.1; 2 = 0.01")
    args = ap.parse_args()
    
    n = int(args.digits)
    p = int(args.precision)
    assert p <= n and p >= 0, "Precision Value cannot be lesser than 0 or \
                              greater than number of digits in the meter."
    
    # Run the inference
    try:
        inference(args.image, args.model, n, p)
    except: raise RuntimeError