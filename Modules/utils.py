import numpy as np

def sum_of_squares(img_channels):
	"""
	Combines complex channels with square root sum of squares.
	"""
	sos =np.sqrt((np.abs(img_channels)**2).sum(axis = -1))
	return sos

def safe_divide(a, b):
	return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def estimate_sensitivity_maps(k_space, acs):
	complex_k_space = k_space[:,:,:,::2]+1j*k_space[:,:,:,1::2]
	acs_k_space = complex_k_space * acs[np.newaxis, :, :, np.newaxis]
	x_hat = np.fft.ifft2(acs_k_space,axes = (1,2))
	norm_factor = np.sqrt(np.sum(np.square(np.abs(x_hat)), axis =(-1))) 
	S = x_hat / norm_factor[:, :, :, np.newaxis]
	S = safe_divide(x_hat, norm_factor[:, :, :, np.newaxis])

	return S

def combine_mc_image(image, S):
	img = np.empty(([*image.shape[0:-1], 2]))

	combined_img = np.sum(np.conj(S) * image, axis =(-1))
	img[:,:,:,::2] = combined_img[:,:,:,np.newaxis].real
	img[:,:,:,1::2] = combined_img[:,:,:,np.newaxis].imag

	return img


def combine_mc_kspace(k_space, S):
	k_masked = np.empty(([*k_space.shape[0:-1], 2]))
	weighted_k = np.empty(k_space.shape)

	image = np.fft.ifft2(k_space[:,:,:,::2]+1j*k_space[:,:,:,1::2],axes = (1,2))

	weighted = np.conj(S) * image
	weighted = np.fft.fft2(weighted, axes = (1,2))
	weighted_k[:,:,:,::2] = weighted[:,:,:,:].real
	weighted_k[:,:,:,1::2] = weighted[:,:,:,:].imag


	combined_image = combine_mc_image(image, S)
	combined_k = np.fft.fft2(combined_image[:,:,:,::2]+1j*combined_image[:,:,:,1::2], axes = (1,2))
	k_masked[:,:,:,::2] = combined_k.real
	k_masked[:,:,:,1::2] = combined_k.imag

	return [k_masked, weighted_k]

