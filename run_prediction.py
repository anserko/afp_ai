import numpy as np
import matplotlib.pyplot as plt

from load_image import load_image
from get_labels import get_labels, get_defect_label
from get_frames import get_frames
from load_models import load_models
from prediction import predict_image, plot_prediction, plot_probabilities, plot_probabilities_2, plot_prediction_single_2


def run_prediction(image_file_name, borders_dict, encoder_pars_dict, lstm_pars_dict, classifier_pars_dict, plot_settings_dict, verbose=0):

	image_data_gt = load_image(image_file_name)

	borders_list = borders_dict[image_file_name]
	image_length = image_data_gt.shape[1]
	defect_type = get_defect_label(image_file_name)
	predicted_data_dict = {}
	labels_dict = {}

	#unpack input data
	stride_step = encoder_pars_dict['stride_step']
	unit_numb_list = encoder_pars_dict['unit_numb_list']
	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	repeat_prediction = lstm_pars_dict['repeat_prediction']
	frames_to_pred_total = lstm_pars_dict['frames_to_pred_total']
	gt_label_mode = lstm_pars_dict['gt_label_mode']
	classifier = classifier_pars_dict['classifier']


	#get ground truth frames
	image_data_frames_gt = get_frames(image_data_gt, stride_step, ifPrint = False)
	#get ground truth labels
	window_size_total = window_size + (window_size_predicted-overlap)*repeat_prediction
	labels_gt = get_labels(
		stride_step,
		borders_list, 
		image_length, 
		window_size_total//gt_label_mode, 
		defect_type
	)[:frames_to_pred_total]
	labels_dict['gt'] = labels_gt

	for unit_numb in unit_numb_list:

		#load models
		models_dict = load_models(unit_numb, stride_step, lstm_pars_dict, classifier=classifier)
		
		image_data_frames_list, image_data_frames_encoded = predict_image(
			image_data_frames_gt, 
			models_dict,
			lstm_pars_dict, 
			repeat_prediction,
			verbose=verbose,
			frames_to_pred_total=frames_to_pred_total
		)
		predicted_data_dict[unit_numb] = image_data_frames_list
		
		#classify
		#probabilities of defects for each frame
		try:
			probabilities = models_dict['model_classifier'].predict(np.array(image_data_frames_encoded), verbose=verbose)
			labels_predicted = np.argmax(probabilities, axis=1)
			labels_dict[unit_numb] = [labels_predicted, probabilities]
		except KeyError:
			#if no classifier is available
			pass	

	return image_data_frames_gt, predicted_data_dict, labels_dict