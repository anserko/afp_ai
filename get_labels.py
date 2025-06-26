import numpy as np
from PIL import Image

from sliding_window import sliding_window
from get_frames import get_frames
from load_image	import load_image

def get_windows_encoded(windows_data_original, model_encoder):
	
	initial_shape = windows_data_original.shape
	window_size = initial_shape[1]
	windows_data_original = windows_data_original.reshape(
		initial_shape[0]*initial_shape[1],
		initial_shape[2],
		initial_shape[3],
		initial_shape[4]
		)
	windows_data_encoded = model_encoder.predict(windows_data_original, verbose=0)
	encoded_shape = windows_data_encoded.shape
	windows_data_encoded = windows_data_encoded.reshape(
		initial_shape[0],
		initial_shape[1],
		encoded_shape[-1]
		)
	return windows_data_encoded

def get_windows_original(image_data_frames, stride_step, window_size, padding_dict):
	
	ifPadding = padding_dict['ifPadding']

	windows_data = np.lib.stride_tricks.sliding_window_view(
		image_data_frames,
		window_shape = window_size,
		axis=0,
		)

	windows_data_original = np.transpose(windows_data, axes=(0,4,1,2,3))

	#padding
	if ifPadding:
		max_window_size = padding_dict['max_window_size']
		pad_value = image_data_frames[0,0,0] #top left pure blue colour array([ 71,  66, 230], dtype=uint8)
		nmb_pad_frames = max_window_size-window_size
		if nmb_pad_frames:
			#in case max_window_size > window_size padding is performed
			original_shape = windows_data_original.shape
			pad_shape = (original_shape[0], nmb_pad_frames, original_shape[2], original_shape[3], original_shape[4])
			pad_image_original = np.full(pad_shape, pad_value)
			windows_data_original = np.append(pad_image_original, windows_data_original, axis=1)

	return windows_data_original



def get_labels(stride_step, borders_list, image_length, window_size, defect_type):

	#number of frames
	frames_nmb = image_length//stride_step
	#get borders for each frame
	frame_borders = np.array([[stride_step*x, stride_step*(x+1)] for x in range(frames_nmb)])
	#get borders for windows
	windows_borders = np.lib.stride_tricks.sliding_window_view(
		frame_borders,
		window_shape = window_size,
		axis=0,
		)
	windows_borders = np.transpose(windows_borders, axes=(0,2,1))
	#get first and last values only
	shape = windows_borders.shape
	windows_borders = windows_borders.reshape(shape[0], shape[1]*shape[2])[:,[0,-1]]
	#get labels
	labels = arrange_labels(windows_borders, borders_list, defect_type)

	return labels

def arrange_labels(windows_borders, defect_borders, defect_type):

	left_border, right_border = defect_borders
	labels = [0 if x[0]>=right_border or x[1]<=left_border else defect_type for x in windows_borders]

	return np.array(labels)

def image_windows_labels(image_name, stride_step, borders_list, lstm_pars_dict, model_encoder, padding_dict=False):

	#load image
	image_data = load_image(image_name)
	defect_type = get_defect_label(image_name)
	#get frames
	image_data_frames = get_frames(image_data, stride_step, ifPrint = False)

	#unpacl lstm settings
	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	repeat_prediction = lstm_pars_dict['repeat_prediction']

	image_length = image_data.shape[1]

	#total size of a window used for classification
	window_size_total = window_size + (window_size_predicted-overlap)*repeat_prediction

	#get labels
	labels = get_labels(stride_step, borders_list, image_length, window_size_total, defect_type)

	#get windows_original
	windows_data = get_windows_original(image_data_frames, stride_step, window_size_total, padding_dict)

	#get windows_encoded
	windows_data_encoded = get_windows_encoded(windows_data, model_encoder)

	return windows_data_encoded, labels

def get_defect_label(image_name):

	defect_type_label_dict = {
	'Twis':1,
	'Puck':2,
	'Fold':3
	}

	try:
		defect_label = defect_type_label_dict[image_name[:4]]
	except KeyError:
		print('Unknown defect type')

	return defect_label

def get_defect_type(label):

	defect_label_type_dict = {
	0:'No defect',
	1:'Twist',
	2:'Puck',
	3:'Fold'
	}

	try:
		defect_type = defect_label_type_dict[label]
	except KeyError:
		defect_type = 'Unknown label'

	return defect_type
