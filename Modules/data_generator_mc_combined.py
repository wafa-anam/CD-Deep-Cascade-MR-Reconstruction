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
		p = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
		X = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
		mask = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
		y1 = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))

		x_hat = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
		S = np.empty((self.batch_size, self.dim[0],self.dim[1], int(self.n_channels / 2)), dtype = 'complex_')
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
		y1[:,:,:,::2] = aux.real
		y1[:,:,:,1::2] = aux.imag

		for i in range(self.batch_size):
			for j in range(self.n_channels):
				x_hat[i,:,:,j] = self.asc * X[i,:,:,j] 
		x_hat = np.fft.ifft2(x_hat[:,:,:,::2]+1j*x_hat[:,:,:,1::2],axes = (1,2))

		X[mask] = 0
		new_var = np.fft.ifft2(X[:,:,:,::2]+1j*X[:,:,:,1::2],axes = (1,2))

		for i in range(self.batch_size):
			abs_sqr = np.square(np.abs(x_hat[i]))
			sqrt_sums =  np.sqrt(np.sum(abs_sqr, axis =(-1)))

			x_ref_combined_i = np.empty((self.dim[0],self.dim[1]), dtype = 'complex_')
			x_ref_combined_i.fill(0)

			x_masked_combined_i = np.empty((self.dim[0],self.dim[1]), dtype = 'complex_')
			x_masked_combined_i.fill(0)

			x_ref_batch = y1[i,:,:,::2] + 1j*y1[i,:,:,1::2]
			x_masked_batch = new_var[i]
			for j in range(x_hat[i].shape[2]):
				S[i,:,:,j] = x_hat[i,:,:,j] / sqrt_sums

				x_ref_combined_i = x_ref_combined_i + ( S[i,:,:,j] * x_ref_batch[:,:,j])
				x_masked_combined_i = x_masked_combined_i + ( S[i,:,:,j] * x_masked_batch[:,:,j])

			x_ref[i, :, :, 0] = x_ref_combined_i.real
			x_ref[i, :, :, 1] = x_ref_combined_i.imag

			k_masked_i = np.fft.fft2(x_masked_combined_i)
			k_masked[i, :, :, 0] = k_masked_i.real
			k_masked[i, :, :, 1] = k_masked_i.imag


		# x_ref = x_ref/self.norm
		y1 = y1/self.norm  # Normalized fully sampled multi-channel reference. Could be converted to root sum of squares.
                      # it depends on how teams model the problem
		X = X/self.norm # Input is the zero-filled reconstruction. Suitable for image-domain methods. Change the code to not 
                   # compute the iFFT if input needs to be in k-space.
		return [k_masked,X,mask,S], x_ref