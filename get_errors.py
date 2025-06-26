import tensorflow as tf
import numpy as np


def get_ssim(image_1, image_2, max_val=255):
	'''
	https://jack-roubaud.medium.com/working-with-structural-similarity-index-86ace619f408
	'''
	#convert images to tensors
	image_1_tensor = tf.convert_to_tensor(
			    image_1,
			    dtype='uint8'
			    )
	image_2_tensor = tf.convert_to_tensor(
		image_2,
		dtype='uint8'
		)

	ssim_value = tf.image.ssim(
		image_1_tensor,
		image_2_tensor,
		max_val=max_val, 
		filter_size=11, 
		filter_sigma=1.5,
		k1=0.01,
		k2=0.03
		)

	return ssim_value.numpy()

def rgb2gray(rgb_image_array) -> np.array:
	'''

	'''
	gray_image_array = np.dot(rgb_image_array[...,:3], [0.2989, 0.5870, 0.1140])
	return gray_image_array


def get_mse(image_1, image_2, axis=None, grayscale=True):
	'''

	'''
	#convert images to grayscale
	if grayscale:
		image_1 = rgb2gray(image_1)
		image_2 = rgb2gray(image_2)

	mse = (np.square(image_1 - image_2)).mean(axis=axis)

	return mse

def get_errors(image_1, image_2):
	'''

	'''
	#define if it's an single image or an array of frames
	if len(image_1.shape)==3:
		#single image
		mse_gray= get_mse(image_1, image_2, grayscale=True)
		mse_rgb= get_mse(image_1, image_2, grayscale=False)
	else:
		#array of frames
		get_mse_rgb = lambda x, y: get_mse(x,y,grayscale=False)
		get_mse_gray = lambda x, y: get_mse(x,y,grayscale=True)
		mse_gray = np.fromiter(map(get_mse_gray, image_1, image_2), dtype='float32')
		mse_rgb = np.fromiter(map(get_mse_rgb, image_1, image_2), dtype='float32')
	
	#calculate for the whole image
	ssim = get_ssim(image_1, image_2)
	
	return (ssim, mse_gray, mse_rgb)
