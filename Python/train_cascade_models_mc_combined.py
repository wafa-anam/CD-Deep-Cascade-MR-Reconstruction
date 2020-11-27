#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Importing standard libraries
import numpy as np
import os
import glob
import sys
import importlib
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf
from skimage.morphology import area_opening

# Include path to my modules
MY_UTILS_PATH = "../Modules/"
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)

# Import my modules
import cs_models_mc_combined as fsnet
from data_generator_mc_combined import DataGenerator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[22]:


# Input parameters
H = 218
W = 170
channels = 12
batch_size = 8
epochs = 100
lr = 1e-4
patience = 10
sampling_mask_path = "../Data/poisson_sampling/R5_218x170.npy"
model_path =  "../Models/model_wwnet_ikik_mcc_r5.h5"
weights_path = "../Models/weights_wwnet_ikik_mcc_r5.h5"
model_string = "ikik"
train_path = "../../data/train_val_12_channel/Train/"
val_path = "../../data/train_val_12_channel/Val/"
cascade = "unet"
nslices = 256
crop = (15, 15)
verbose = True


# In[23]:


train = glob.glob(train_path + "*.h5")
val = glob.glob(val_path + "*.h5")

# undersampling patterns - centered k-space - 100 pre computed patterns
var_sampling_mask = np.load(sampling_mask_path)

asc = area_opening(var_sampling_mask[0], area_threshold=10, connectivity=1, parent=None, tree_traverser=None)


if verbose:
    print(cascade)
    print("Domains: ", model_string)
    print("Model name: ", model_path)
    print("Weights path: ", weights_path)
    print("Sampling:", 1.0*var_sampling_mask.sum() / (var_sampling_mask.size*0.85)) # 85 % sampling Kz direction
    print("Train path: ", train_path)
    print("Validation path: ", val_path)


# Replicate sampling mask across number of channels
var_sampling_mask = np.repeat(var_sampling_mask[:, :, :, np.newaxis], 24, axis = -1) 

train_generator = DataGenerator(train, dim = (H, W), under_masks = var_sampling_mask,  crop = crop, batch_size = batch_size, n_channels = channels * 2, nslices = nslices, shuffle=True, asc = asc)

val_generator = DataGenerator(val, dim = (H, W), under_masks = var_sampling_mask,  crop = crop, batch_size = batch_size, n_channels = channels * 2, nslices= nslices, shuffle=False, asc = asc)



# Callbacks
earlyStopping = EarlyStopping(monitor='val_loss',
                                           patience=patience, 
                                           verbose=0, mode='min')

checkpoint = ModelCheckpoint(weights_path, mode = 'min',                              monitor='val_loss',verbose=0,                             save_best_only=True, save_weights_only = True)

checkpoint2 = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False,                save_weights_only=False, mode='min')

Hpad = (8 - (H%8)) // 2
Wpad = (8 - (W%8)) // 2
if os.path.isfile(model_path):
    model = load_model(model_path)
elif  cascade == 'unet':
    model = fsnet.deep_cascade_unet(depth_str = model_string, H = H, W = W, Hpad = Hpad, Wpad = Wpad, channels = channels) 
elif cascade == 'flat':
    model = fsnet.deep_cascade_flat_unrolled(depth_str = model_string, H = H,W = W,depth = 6, kshape = (3,3), nf = 64,channels = channels)
else:
   pass

if not os.path.isfile(model_path):
    opt = Adam(learning_rate = lr)
    model.compile(loss = 'mse',optimizer=opt)

if verbose:
    print(model.summary())

hist = model.fit(train_generator, epochs = epochs, steps_per_epoch = ( (nslices - crop[0] - crop[1])*len(train) ) // batch_size,              validation_data = val_generator, validation_steps = ( (nslices - crop[0] - crop[1])*len(val) ) // batch_size,                    callbacks = [ checkpoint2, checkpoint, earlyStopping ], verbose = 1)

# Save model history
with open(model_path[:-2] + "history", 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)



# In[ ]:




