#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pylab as plt
import os
import glob
import sys
import natsort
from tensorflow.keras.optimizers import  Adam
import nibabel as nib
import h5py
import tensorflow as tf

# Include path to my modules
MY_UTILS_PATH = "../Modules/"
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)

# Import my modules
import cs_models_mc_combined as fsnet
import utils

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[2]:


# Input parameters
channels = 12
batch_size = 8
weights_path = "../Models/weights_wwnet_ikik_mcc_r5.h5"
model_string = "ikik"
test_path = "../../data/test_12_channel/Test-R=5/"
results_path = "../WWnet-ikik-combined/12-channel-R=5/"
cascade = "unet"
crop = (50,-50) # slices to crop
verbose = True


# In[3]:


test_files = np.asarray(glob.glob(test_path + "*.h5"))

# Sort by kx - ky size
files_sizes = []
for ii in test_files:
    with h5py.File(ii, 'r') as f:
        kspace_test = f['kspace']
        files_sizes.append(kspace_test.shape[1]*kspace_test.shape[2]) 
files_sizes = np.asarray(files_sizes)
indexes = np.argsort(files_sizes)
test_files = test_files[indexes]

if verbose:
    print(cascade)
    print("Domains: ", model_string)
    print("Number of files:", len(test_files))
    print(test_files[0])
    print("Weights path:", weights_path)
    print("Test path:", test_path)
    print("results path:", results_path)
    
norm = np.sqrt(218*170)
model_exists = False
for ii in test_files:

    name = ii.split("/")[-1]
    # Load data
    with h5py.File(ii, 'r') as f:
        kspace_test=  np.array(f.get('kspace')).astype(np.float32)[crop[0]:crop[1]]
    
    kspace_test = kspace_test/norm

    Z,H,W,_ = kspace_test.shape
    if H*W != (218*170): 
        model_exists = False

    Wpad = (8 -W%8)//2
    Hpad = (8 -H%8)//2

    if not model_exists:
        if cascade == 'unet':
            model = fsnet.deep_cascade_unet(depth_str = model_string, H = H, W = W, Hpad = Hpad, Wpad = Wpad, channels = channels) 
        elif cascade == 'flat':
            model = fsnet.deep_cascade_flat_unrolled(depth_str = model_string, H = H,W = W,depth = 14, kshape = (3,3), nf = 116,channels = channels)              

        model_exists = True
        opt = Adam()
        model.compile(loss = 'mse',optimizer = opt)
        model.load_weights(weights_path)   

    var_sampling_mask = ((kspace_test == 0)).astype(np.float32)
    S = utils.estimate_sensitivity_maps(kspace_test)
    [k_space, weighted_k_space] = utils.combine_mc_kspace(kspace_test, S)
    pred = model.predict([ k_space, weighted_k_space, var_sampling_mask, S],batch_size = batch_size)*norm
    pred = pred[:,:,:,::2]+1j*pred[:,:,:,1::2]
    pred = np.sqrt((np.abs(pred)**2).sum(axis = -1)) # Root sum of squares

    if verbose:
        print(os.path.join(results_path,name))
    with h5py.File(os.path.join(results_path,name), 'w') as hf:
        hf.create_dataset('reconstruction', data=pred)

