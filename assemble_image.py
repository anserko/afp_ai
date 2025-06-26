import numpy as np


def assemble_image(image_data_frames, to_int=True):
	'''

	'''
	#reshape from separate frames into a single image
	original_shape = image_data_frames.shape
	#swap axis
	image_data_rsh1 = np.swapaxes(image_data_frames,1,2)
	#reshape
	image_data_rsh2 = image_data_rsh1.reshape(original_shape[0]*original_shape[2], original_shape[1], original_shape[3])
	#swap axis back
	image_data = np.swapaxes(image_data_rsh2,0,1)
	if to_int:
		#convert float to int
		image_data=image_data.astype(int)

	return image_data
