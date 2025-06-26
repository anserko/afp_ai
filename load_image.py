import numpy as np
from PIL import Image

def load_image(filename):
	dtype = 'uint8'
	#get frames from the image
	path_list = ['PuckerImages/RGB_cropped',
	             'TwistImages/RGB',
	             'FoldImages/RGB',]

	for path in path_list:
	    if filename[:4]==path[:4]:
	        break        
	img = Image.open(f'{path}/{filename}')
	image_data = np.array(img, dtype=dtype)

	image_data = image_data[:,:-96,:]
	return image_data