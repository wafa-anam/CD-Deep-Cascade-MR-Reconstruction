import numpy as np
import h5py
import os
from tensorflow import keras
from skimage.morphology import area_opening

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, dim, under_masks, crop, batch_size, asc, n_channels,nslices = 256, shuffle=True):
		self.list_IDs = list_IDs
		self.dim = dim
		self.under_masks = ~under_masks
		self.crop = crop # Remove slices with no or little anatomy
		self.batch_size = batch_size
		self.n_channels = n_channels
		self.nslices = nslices
		self.shuffle = shuffle
		self.nsamples = len(self.list_IDs)*(self.nslices - self.crop[0] - self.crop[1])
		self.on_epoch_end()
		self.norm = np.sqrt(dim[0]*dim[1])
		self.asc = asc
	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(self.nsamples/ self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'

	# Generate indexes of the batch
		batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, Y = self.__data_generation(batch_indexes)

		return X, Y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(self.nsamples)
		if self.shuffle == True:
		    np.random.shuffle(self.indexes)

	def __data_generation(self, batch_indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
		mask = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))

		x_ref = np.empty((self.batch_size, self.dim[0],self.dim[1], 2))
		k_masked = np.empty((self.batch_size, self.dim[0],self.dim[1], 2))

		if self.shuffle:
		    idxs = np.random.choice(np.arange(self.under_masks.shape[0], dtype=int), self.batch_size, replace = True)
		else:
		    idxs = np.arange(0,self.batch_size,dtype = int)
		mask = self.under_masks[idxs]

		# Generate data
		for ii in range(batch_indexes.shape[0]):
		    # Store sample
			file_id = batch_indexes[ii]//(self.nslices - self.crop[0] - self.crop[1])
			file_slice = batch_indexes[ii]%(self.nslices - self.crop[0] - self.crop[1])
			# Load data
			with h5py.File(self.list_IDs[file_id], 'r') as f:
				kspace = f['kspace']
                # Most volumes have 170 slices, but some have more. For these cases we crop back to 170 during training.
                # Could be made more generic.
				if kspace.shape[2] == self.dim[1]:
					X[ii,:,:,:] = kspace[self.crop[0]+file_slice]

				else:
					idx = int((kspace.shape[2] - self.dim[1])/2)
					X[ii,:,:,:] = kspace[self.crop[0]+file_slice,:,idx:-idx,:]
		aux = np.fft.ifft2(X[:,:,:,::2]+1j*X[:,:,:,1::2],axes = (1,2))
		
		# estimate sensititvity maps
		complex_k_space = X[:,:,:,::2]+1j*X[:,:,:,1::2]
		acs_k_space = complex_k_space * self.asc[np.newaxis, :, :, np.newaxis]
		x_hat = np.fft.ifft2(acs_k_space,axes = (1,2))
		norm_factor = np.sqrt(np.sum(np.square(np.abs(x_hat)), axis =(-1))) 
		#TODO ask about this (from data/transforms.py root sum of squares)
		# norm_factor = np.sqrt(np.sum(x_hat.real **2 + x_hat.imag ** 2, axis =(-1))) 
		S = x_hat / norm_factor[:, :, :, np.newaxis]
		S = safe_divide(x_hat, norm_factor[:, :, :, np.newaxis])

		# mask k-space
		X[mask] = 0
		masked_image = np.fft.ifft2(X[:,:,:,::2]+1j*X[:,:,:,1::2],axes = (1,2))

		# use sensitivities to combine masked data
		combined_masked_img = np.sum(np.conj(S) * masked_image, axis =(-1))
		combined_masked_k = np.fft.fft2(combined_masked_img, axes = (1,2))
		k_masked[:,:,:,::2] = combined_masked_k[:,:,:,np.newaxis].real
		k_masked[:,:,:,1::2] = combined_masked_k[:,:,:,np.newaxis].imag

		# use sensitivities to combine reference
		combined_ref = np.sum(np.conj(S) * aux, axis =(-1))
		x_ref[:,:,:,::2] = combined_ref[:,:,:,np.newaxis].real
		x_ref[:,:,:,1::2] = combined_ref[:,:,:,np.newaxis].imag


		X = X/self.norm # Input is the zero-filled reconstruction. Suitable for image-domain methods. Change the code to not 
                   # compute the iFFT if input needs to be in k-space.
		
		return [k_masked,X,mask,S], x_ref

def safe_divide(a, b):
	return np.divide(a, b, out=np.zeros_like(a), where=b!=0)