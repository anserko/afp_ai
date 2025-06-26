import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from assemble_image import assemble_image
from get_errors import get_errors
from get_labels import get_defect_type, get_windows_original, get_windows_encoded
from load_image import load_image
from get_frames import get_frames
import time


def predict_step(image_data_frames_input, models_dict, lstm_pars_dict, repeat_prediction=1, verbose=0):

	#get lstm settings
	stateful = lstm_pars_dict['stateful']
	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']

	#get models
	model_encoder = models_dict['model_encoder']
	model_decoder = models_dict['model_decoder']
	model_lstm = models_dict['model_lstm']

	#encoding input window
	image_data_frames_encoded = model_encoder.predict(image_data_frames_input, verbose=verbose)

	image_data_frames_lstm = np.copy(image_data_frames_encoded)
	lstm_input = image_data_frames_lstm

	for i in range(repeat_prediction):
		#reset lstm state to get independent prediction between timesteps (batches)
		#model_lstm.reset_states()
		predicted_sequence = model_lstm.predict(lstm_input[np.newaxis], verbose=verbose)
		if window_size_predicted==1:
			predicted_sequence_append = predicted_sequence
		else:
			predicted_sequence_append = predicted_sequence[0][overlap:]
		lstm_input = np.concatenate((lstm_input[window_size_predicted-overlap:],predicted_sequence_append), axis=0)
		image_data_frames_lstm = np.concatenate((image_data_frames_lstm, predicted_sequence_append), axis=0)

	#cut out ground truth
	image_data_frames_lstm = image_data_frames_lstm[window_size:]

	#decode output data
	image_data_frames_decoded = model_decoder.predict(image_data_frames_lstm, verbose=verbose)
	
	#encoded input+output for classifier
	image_data_frames_encoded_classify = np.concatenate((image_data_frames_encoded, image_data_frames_lstm), axis=0)

	return image_data_frames_decoded, image_data_frames_encoded_classify


def predict_image(image_data_frames_gt,models_dict,lstm_pars_dict,repeat_prediction,verbose=0,frames_to_pred_total=False):
	
	image_data_frames_list = []
	image_data_frames_encoded_list = []
	window_size = lstm_pars_dict['window_size']

	number_of_frames = image_data_frames_gt.shape[0]
	if not frames_to_pred_total:
		frames_to_pred_total = number_of_frames - repeat_prediction-1

	for i in range(frames_to_pred_total):
		#get ground truth window
		image_data_frames_input = image_data_frames_gt[i:window_size+i]

		image_data_frames_decoded, image_data_frames_encoded_classify = predict_step(
			image_data_frames_input,
			models_dict,
			lstm_pars_dict,
			repeat_prediction=repeat_prediction, 
			verbose=verbose
			)
		image_data_frames_list.append(image_data_frames_decoded)
		image_data_frames_encoded_list.append(image_data_frames_encoded_classify)

	return image_data_frames_list, image_data_frames_encoded_list



def classify_image_frames(filename, stride_step, window_size_total, padding_dict, borders_dict, models_dict):

	image_data = load_image(filename)
	image_data_frames = get_frames(image_data, stride_step, ifPrint = False)
	windows_data = get_windows_original(image_data_frames, stride_step, window_size_total, padding_dict)
	windows_data_encoded = get_windows_encoded(windows_data, models_dict['model_encoder'])
	labels_probs = models_dict['model_classifier'].predict(windows_data_encoded, verbose=0)  
	# Convert probabilities to class labels
	labels = np.argmax(labels_probs, axis=1)

	return labels










def predict_step_2(image_data_frames_input, models_dict, lstm_pars_dict, repeat_prediction=1, verbose=0):

	#get lstm settings
	stateful = lstm_pars_dict['stateful']
	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']

	#get models
	model_encoder = models_dict['model_encoder']
	model_decoder = models_dict['model_decoder']
	model_lstm = models_dict['model_lstm']

	#encoding input window
	image_data_frames_encoded = model_encoder.predict(image_data_frames_input, verbose=verbose)
	initial_input_shape = image_data_frames_encoded.shape

	for i in range(repeat_prediction):
		#slide windows
		lstm_input = np.lib.stride_tricks.sliding_window_view(
			image_data_frames_encoded,
			window_shape = window_size,
			axis=0,
			)
		lstm_input = np.transpose(lstm_input, axes=(0,2,1))

		predicted_sequence = model_lstm.predict(lstm_input, verbose=verbose)

		if window_size_predicted==1:
			predicted_sequence_append = predicted_sequence[-1:]
		else:
			predicted_sequence_append = predicted_sequence[-1][overlap:]

		image_data_frames_encoded = np.concatenate((image_data_frames_encoded, predicted_sequence_append),axis=0)


	#cut out ground truth
	image_data_frames_encoded = image_data_frames_encoded[initial_input_shape[0]:]

	#decode output data
	image_data_frames_decoded = model_decoder.predict(image_data_frames_encoded, verbose=verbose)

	return image_data_frames_decoded


def predict_image_2(image_data_frames_gt,models_dict,lstm_pars_dict,repeat_prediction,verbose=0,frames_to_pred_total=False):
	
	image_data_frames_list = []
	window_size = lstm_pars_dict['window_size']

	number_of_frames = image_data_frames_gt.shape[0]
	if not frames_to_pred_total:
		frames_to_pred_total = number_of_frames - repeat_prediction-1

	for i in range(frames_to_pred_total):
		#get ground truth window
		image_data_frames_input = image_data_frames_gt[:window_size+i]

		image_data_frames_decoded = predict_step_2(
			image_data_frames_input,
			models_dict,
			lstm_pars_dict,
			repeat_prediction=repeat_prediction, 
			verbose=verbose
			)
		image_data_frames_list.append(image_data_frames_decoded)

	return image_data_frames_list




def plot_prediction(image_data_frames_gt,predicted_data_dict,lstm_pars_dict,repeat_prediction=1,labels_dict=False,numb_cols=3):
	
	unit_numb_list = list(predicted_data_dict.keys())
	predicted_data_list = [predicted_data_dict[x] for x in unit_numb_list]

	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	stride_step = image_data_frames_gt.shape[2]

	frames_to_pred_total = len(predicted_data_list[0])
	image_data_frames_gt_output_list = []

	#get pairs ground truth prediction
	for i in range(frames_to_pred_total):
		
		index_start = window_size+i
		index_end = index_start+(window_size_predicted-overlap)*repeat_prediction
		image_data_frames_gt_output = image_data_frames_gt[index_start:index_end]
		image_data_frames_gt_output_list.append(image_data_frames_gt_output)
		
	frames_to_plot = list(zip(image_data_frames_gt_output_list,[*zip(*predicted_data_list)]))

	numb_cols = numb_cols
	#all predicted plots + ground truth + errors
	numb_plots_per_frame = len(predicted_data_list)+2

	number_of_frames = len(frames_to_plot)
	numb_rows = int(np.ceil(number_of_frames/numb_cols))

	fig = plt.figure(figsize=(20, 3.7*numb_plots_per_frame*numb_rows))
	outer = gridspec.GridSpec(numb_rows, numb_cols, wspace=0.2, hspace=0.1)
	linewidth = 3.0
	fontsize_sec = 14

	#assemble gt image
	image_data_gt = assemble_image(image_data_frames_gt)

	ssim_frames_list = []
	mse_gray_frames_list = []
	mse_rgb_frames_list = []

	for i, frames in enumerate(frames_to_plot):
		frames_gt = frames[0]
		frames_prediction_list = frames[1]
		#plot
		heights = [1]*numb_plots_per_frame
		heights[-1] = 0.5

		inner = gridspec.GridSpecFromSubplotSpec(
			numb_plots_per_frame, 1,
			subplot_spec=outer[i], hspace=0.005, height_ratios=heights)

		#plot ground truth
		ax = plt.Subplot(fig, inner[0])
		x_lim = image_data_gt.shape[1]
		#plot only a portion of gt for comparison
		index_current = stride_step*(window_size+i+(window_size_predicted-overlap)*repeat_prediction)
		ax.imshow(image_data_gt[:,:index_current,:])
		ax.set_xlim(0, x_lim)
		if labels_dict:
			defect_label_gt = labels_dict['gt'][i]
			defect_type_gt = get_defect_type(defect_label_gt)
		else:
			defect_type_gt = 'No label'
		ax.set_title(f'Step {i+1}, Ground truth. {defect_type_gt}')
		#plot separator input/prediction
		x_separator = stride_step*(window_size+i)
		ax.axvline(x = x_separator, color = 'red')
		#add window borders
		x_window_left = x_separator-stride_step*window_size
		ax.axvline(x = x_window_left, color = 'orange')
		x_window_right = x_separator+stride_step*((window_size_predicted-overlap)*repeat_prediction)
		ax.axvline(x = x_window_right, color = 'orange')
		fig.add_subplot(ax)

		for ii, frames_prediction in enumerate(frames_prediction_list):
			unit_numb = unit_numb_list[ii]
			if labels_dict:
				defect_label_predicted = labels_dict[unit_numb][0][i]
				defect_type_predicted = get_defect_type(defect_label_predicted)
			else:
				defect_type_predicted = 'No label'
			image_data_predicted = assemble_image(frames_prediction)
			#plot prediction
			ax = plt.Subplot(fig, inner[ii+1])
			#reassemble to include all previous gt
			image_data_plot = np.concatenate(
				(
					image_data_gt[:,:x_separator,:],
					image_data_predicted
				),
				axis=1
			)
			ax.imshow(image_data_plot)
			ax.set_title(f'Prediction. LS = {unit_numb}. {defect_type_predicted}')
			ax.set_xlim(0, x_lim)
			ax.axvline(x = x_separator, color = 'red')
			ax.axvline(x = x_window_left, color = 'orange')
			ax.axvline(x = x_window_right, color = 'orange')
			fig.add_subplot(ax)
		
		if 1:

			inner_error = gridspec.GridSpecFromSubplotSpec(
						1, 3,
						subplot_spec=inner[-1], wspace=0.4
						)

			#plot errors
			ssim_frames_list_temp = []
			mse_gray_frames_list_temp = []
			mse_rgb_frames_list_temp = []
			ax_mse_rgb = plt.Subplot(fig, inner_error[0])
			ax_mse_rgb.set_title(f'MSE RGB') 
			ax_mse_gray = plt.Subplot(fig, inner_error[1])
			ax_mse_gray.set_title(f'MSE Gray') 
			ax_ssim = plt.Subplot(fig, inner_error[2])
			ax_ssim.set_title(f'SSIM')  

			for ii, frames_prediction in enumerate(frames_prediction_list):

				ssim_frames, mse_gray_frames, mse_rgb_frames = get_errors(
					frames_gt, 
					frames_prediction
						)
				ssim_frames_list_temp.append(ssim_frames)
				mse_gray_frames_list_temp.append(mse_gray_frames)
				mse_rgb_frames_list_temp.append(mse_rgb_frames)
				
				ax_mse_gray.plot(mse_gray_frames, label=f'ls: {unit_numb_list[ii]}')
				ax_mse_rgb.plot(mse_rgb_frames, label=f'ls: {unit_numb_list[ii]}')			
				ax_ssim.plot(ssim_frames, label=f'ls: {unit_numb_list[ii]}')

			ax_mse_rgb.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			ax_mse_gray.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			ax_ssim.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			x_lim = len(mse_gray_frames)
			ax_mse_rgb.set_xlim(0, x_lim)
			ax_mse_gray.set_xlim(0, x_lim)
			ax_ssim.set_xlim(0, x_lim)
			fig.add_subplot(ax_mse_rgb)
			fig.add_subplot(ax_mse_gray)
			fig.add_subplot(ax_ssim)

				
			ssim_frames_list.append(ssim_frames_list_temp)
			mse_gray_frames_list.append(mse_gray_frames_list_temp)
			mse_rgb_frames_list.append(mse_rgb_frames_list_temp)


def plot_probabilities(labels_dict, window_total, stride_step, image_name):

	image_data = load_image(image_name)
	image_length = image_data.shape[1]

	fig, axes = plt.subplots(2+(len(labels_dict)-1),1, figsize=(20,20+10*(len(labels_dict)-1)))
	linewidth = 3.0

	fontsize_sec = 14
	ax = axes[0]
	ax.imshow(image_data, aspect="auto")

	ax = axes[1]
	labels_gt = labels_dict['gt']
	x_length = labels_gt.shape[0]
	x = np.linspace(window_total*stride_step, (x_length+window_total-1)*stride_step, x_length)
	ax.plot(x, labels_gt, marker=8, label='Ground truth')
	for key in labels_dict.keys():
		if type(key)!=int:
			#skip gt key
			continue
		labels_predicted = labels_dict[key][0]
		ax.plot(x, labels_predicted, marker=8, label=f'LS: {key}')
	ax.legend(fontsize=fontsize_sec+4, loc='upper left')
	ax.set_title(f'Defect detection', fontsize=fontsize_sec+5, fontweight='bold') 
	ax.set_xlabel('Defect_type', fontsize=fontsize_sec+2, fontweight='bold')
	ax.set_ylabel('Image length', fontsize=fontsize_sec+2, fontweight='bold')
	ax.tick_params(axis='both', labelsize=fontsize_sec+2)
	ax.set_xlim(0, image_length)

	print(axes.shape)
	for i, key in enumerate(labels_dict.keys()):	
		if type(key)!=int:
			#skip gt key
			continue
		ax = axes[1+i]
		probabilities_predicted = labels_dict[key][1]
		for ii in range(probabilities_predicted.shape[1]):
			probability = probabilities_predicted[:,ii]
			ax.plot(x, probability, marker=8, label=get_defect_type(ii))
		ax.legend(fontsize=fontsize_sec+4, loc='upper left')
		ax.set_title(f'Probabilities. LS: {key}', fontsize=fontsize_sec+5, fontweight='bold') 
		ax.set_xlabel('Defect_probability', fontsize=fontsize_sec+2, fontweight='bold')
		ax.set_ylabel('Image length', fontsize=fontsize_sec+2, fontweight='bold')
		ax.tick_params(axis='both', labelsize=fontsize_sec+2)
		ax.set_xlim(0, image_length)

def plot_probabilities_2(labels_dict, window_total, stride_step, image_name):

	image_data = load_image(image_name)
	image_length = image_data.shape[1]

	fig, axes = plt.subplots(2+(len(labels_dict)-1),1, figsize=(20,20+10*(len(labels_dict)-1)))
	linewidth = 3.0

	fontsize_sec = 14
	fontsize_title = fontsize_sec+12
	fontsize_label = fontsize_sec+6
	fontsize_legend = fontsize_sec+6
	marker = 'h'
	markersize = 8
	ax = axes[0]
	ax.imshow(image_data, aspect="auto")
	ax.set_yticks([])
	ax.tick_params(axis='x', labelsize=fontsize_label)


	ax = axes[1]
	labels_gt = labels_dict['gt']
	x_length = labels_gt.shape[0]
	x = np.linspace(window_total*stride_step, (x_length+window_total-1)*stride_step, x_length)
	ax.plot(x, labels_gt, linewidth=linewidth, marker=marker, markersize=markersize, label='Ground truth')
	for key in labels_dict.keys():
		if type(key)!=int:
			#skip gt key
			continue
		labels_predicted = labels_dict[key][0]
		ax.plot(x, labels_predicted, linewidth=linewidth, marker=marker, markersize=markersize, label=f'LS: {key}')
	ax.legend(fontsize=fontsize_legend, loc='upper left')
	ax.set_title(f'Defect detection', fontsize=fontsize_title, fontweight='bold') 
	ax.set_xlabel('Image length', fontsize=fontsize_label, fontweight='bold')
	ax.set_ylabel('Defect_type probability', fontsize=fontsize_label, fontweight='bold')	
	ax.set_xlim(0, image_length)

	ax.set_yticks(
				np.arange(0, 4, step=1),
				labels=[get_defect_type(x) for x in range(4)],
				)
	ax.tick_params(axis='both', labelsize=fontsize_label)

	print(axes.shape)
	for i, key in enumerate(labels_dict.keys()):	
		if type(key)!=int:
			#skip gt key
			continue
		ax = axes[1+i]
		probabilities_predicted = labels_dict[key][1]
		for ii in range(probabilities_predicted.shape[1]):
			probability = probabilities_predicted[:,ii]
			ax.plot(x, probability, linewidth=linewidth, marker=marker, markersize=markersize, label=get_defect_type(ii))
		ax.legend(fontsize=fontsize_legend, loc='center left')
		ax.set_title(f'Defect_type probabilities. LS: {key}', fontsize=fontsize_title, fontweight='bold') 
		ax.set_xlabel('Image length', fontsize=fontsize_label, fontweight='bold')
		ax.set_ylabel('Defect_probability', fontsize=fontsize_label, fontweight='bold')
		ax.tick_params(axis='both', labelsize=fontsize_label)
		ax.set_xlim(0, image_length)


def plot_prediction_single(image_data_frames_gt,predicted_data_dict,lstm_pars_dict,repeat_prediction,plot_settings_dict):
	
	labels_dict = plot_settings_dict['labels_dict']
	ifMetrics = plot_settings_dict['ifMetrics']
	ifSave = plot_settings_dict['ifSave']
	save_folder = plot_settings_dict['save_folder']

	unit_numb_list = list(predicted_data_dict.keys())
	predicted_data_list = [predicted_data_dict[x] for x in unit_numb_list]

	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	stride_step = image_data_frames_gt.shape[2]

	frames_to_pred_total = len(predicted_data_list[0])
	image_data_frames_gt_output_list = []

	#get pairs ground truth prediction
	for i in range(frames_to_pred_total):
		
		index_start = window_size+i
		index_end = index_start+(window_size_predicted-overlap)*repeat_prediction
		image_data_frames_gt_output = image_data_frames_gt[index_start:index_end]
		image_data_frames_gt_output_list.append(image_data_frames_gt_output)
		
	frames_to_plot = list(zip(image_data_frames_gt_output_list,[*zip(*predicted_data_list)]))

	#all predicted plots + ground truth + metrics/errors
	numb_plots_per_frame = len(predicted_data_list)+1
	if ifMetrics:
		numb_plots_per_frame = numb_plots_per_frame+1

	number_of_frames = len(frames_to_plot)

	#assemble gt image
	image_data_gt = assemble_image(image_data_frames_gt)

	ssim_frames_list = []
	mse_gray_frames_list = []
	mse_rgb_frames_list = []

	for i, frames in enumerate(frames_to_plot):

		fig = plt.figure(figsize=(7,3.7*numb_plots_per_frame))
		#plot
		heights = [1]*numb_plots_per_frame
		if ifMetrics:
			#metrics plots are half the height of prediction plot
			heights[-1] = 0.5

		
		inner = gridspec.GridSpec(numb_plots_per_frame, 1, hspace=0.25, height_ratios=heights)

		linewidth = 2
		fontsize_sec = 14

		fig.suptitle(f'Step {i+1}',fontsize=fontsize_sec+4, fontweight='bold', y=0.93)

		frames_gt = frames[0]
		frames_prediction_list = frames[1]
		#plot
		#plot ground truth
		ax = plt.Subplot(fig, inner[0])
		x_lim = image_data_gt.shape[1]
		#plot only a portion of gt for comparison
		index_current = stride_step*(window_size+i+(window_size_predicted-overlap)*repeat_prediction)
		ax.imshow(image_data_gt[:,:index_current,:])
		ax.set_xlim(0, x_lim)
		if labels_dict:
			defect_label_gt = labels_dict['gt'][i]
			defect_type_gt = get_defect_type(defect_label_gt)
		else:
			defect_type_gt = 'No label'
		if defect_type_gt=='Puck':
			defect_type_gt='Pucker'
		ax.set_title(f'Ground truth. {defect_type_gt}', fontweight='bold',fontsize=fontsize_sec+2)
		#plot separator input/prediction
		x_separator = stride_step*(window_size+i)
		ax.axvline(x = x_separator, color = 'red',linewidth=linewidth)
		#add window borders
		x_window_left = x_separator-stride_step*window_size
		ax.axvline(x = x_window_left, color = 'orange',linewidth=linewidth)
		x_window_right = x_separator+stride_step*((window_size_predicted-overlap)*repeat_prediction)
		ax.axvline(x = x_window_right, color = 'orange',linewidth=linewidth)
		fig.add_subplot(ax)

		for ii, frames_prediction in enumerate(frames_prediction_list):
			unit_numb = unit_numb_list[ii]
			if labels_dict:
				defect_label_predicted = labels_dict[unit_numb][0][i]
				defect_type_predicted = get_defect_type(defect_label_predicted)
			else:
				defect_type_predicted = 'No label'
			image_data_predicted = assemble_image(frames_prediction)
			#plot prediction
			ax = plt.Subplot(fig, inner[ii+1])
			#reassemble to include all previous gt
			image_data_plot = np.concatenate(
				(
					image_data_gt[:,:x_separator,:],
					image_data_predicted
				),
				axis=1
			)
			ax.imshow(image_data_plot)
			if defect_type_predicted=='Puck':
				defect_type_predicted='Pucker'
			ax.set_title(f'Prediction. LS = {unit_numb}. {defect_type_predicted}',fontsize=fontsize_sec+2, fontweight='bold')
			ax.set_xlim(0, x_lim)
			ax.axvline(x = x_separator, color = 'red',linewidth=linewidth)
			ax.axvline(x = x_window_left, color = 'orange',linewidth=linewidth)
			ax.axvline(x = x_window_right, color = 'orange',linewidth=linewidth)
			fig.add_subplot(ax)

		
		if ifMetrics:

			inner_error = gridspec.GridSpecFromSubplotSpec(
						1, 3,
						subplot_spec=inner[-1], wspace=0.35
						)


			#plot errors
			ssim_frames_list_temp = []
			mse_gray_frames_list_temp = []
			mse_rgb_frames_list_temp = []
			if 1:
				ax_mse_rgb = plt.Subplot(fig, inner_error[0])
				ax_mse_gray = plt.Subplot(fig, inner_error[1])
				ax_ssim = plt.Subplot(fig, inner_error[2])
			ax_mse_rgb.set_title(f'MSE RGB') 
			ax_mse_gray.set_title(f'MSE Gray') 
			ax_ssim.set_title(f'SSIM')  

			for ii, frames_prediction in enumerate(frames_prediction_list):

				ssim_frames, mse_gray_frames, mse_rgb_frames = get_errors(
					frames_gt, 
					frames_prediction
						)
				ssim_frames_list_temp.append(ssim_frames)
				mse_gray_frames_list_temp.append(mse_gray_frames)
				mse_rgb_frames_list_temp.append(mse_rgb_frames)
				
				ax_mse_gray.plot(mse_gray_frames, label=f'ls: {unit_numb_list[ii]}')
				ax_mse_rgb.plot(mse_rgb_frames, label=f'ls: {unit_numb_list[ii]}')			
				ax_ssim.plot(ssim_frames, label=f'ls: {unit_numb_list[ii]}')

			ax_mse_rgb.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			ax_mse_gray.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			ax_ssim.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			x_lim = len(mse_gray_frames)
			ax_mse_rgb.set_xlim(0, x_lim)
			ax_mse_gray.set_xlim(0, x_lim)
			ax_ssim.set_xlim(0, x_lim)
			if 1:
				fig.add_subplot(ax_mse_rgb)
				fig.add_subplot(ax_mse_gray)
				fig.add_subplot(ax_ssim)

				
			ssim_frames_list.append(ssim_frames_list_temp)
			mse_gray_frames_list.append(mse_gray_frames_list_temp)
			mse_rgb_frames_list.append(mse_rgb_frames_list_temp)

		if ifSave:
			filename = f'{save_folder}/step_{i+1}.svg'
			plt.savefig(filename, bbox_inches='tight')




def plot_prediction_single_2(image_data_frames_gt,predicted_data_dict,lstm_pars_dict,repeat_prediction,plot_settings_dict):
	
	#exclude MSE Gray metric

	labels_dict = plot_settings_dict['labels_dict']
	ifMetrics = plot_settings_dict['ifMetrics']
	ifSave = plot_settings_dict['ifSave']
	save_folder = plot_settings_dict['save_folder']
	ifClose = plot_settings_dict['ifClose']

	unit_numb_list = list(predicted_data_dict.keys())
	predicted_data_list = [predicted_data_dict[x] for x in unit_numb_list]

	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	stride_step = image_data_frames_gt.shape[2]

	frames_to_pred_total = len(predicted_data_list[0])
	image_data_frames_gt_output_list = []

	#get pairs ground truth prediction
	for i in range(frames_to_pred_total):
		
		index_start = window_size+i
		index_end = index_start+(window_size_predicted-overlap)*repeat_prediction
		image_data_frames_gt_output = image_data_frames_gt[index_start:index_end]
		image_data_frames_gt_output_list.append(image_data_frames_gt_output)
		
	frames_to_plot = list(zip(image_data_frames_gt_output_list,[*zip(*predicted_data_list)]))

	#all predicted plots + ground truth + metrics/errors
	numb_plots_per_frame = len(predicted_data_list)+1
	if ifMetrics:
		numb_plots_per_frame = numb_plots_per_frame+1

	number_of_frames = len(frames_to_plot)

	#assemble gt image
	image_data_gt = assemble_image(image_data_frames_gt)

	ssim_frames_list = []
	mse_gray_frames_list = []
	mse_rgb_frames_list = []

	for i, frames in enumerate(frames_to_plot):

		fig = plt.figure(figsize=(7,3.7*numb_plots_per_frame))
		#plot
		heights = [1]*numb_plots_per_frame
		if ifMetrics:
			#metrics plots are half the height of prediction plot
			heights[-1] = 0.5

		
		inner = gridspec.GridSpec(numb_plots_per_frame, 1, hspace=0.25, height_ratios=heights)

		linewidth = 2
		fontsize_sec = 14

		fig.suptitle(f'Step {i+1}',fontsize=fontsize_sec+4, fontweight='bold', y=0.92)

		frames_gt = frames[0]
		frames_prediction_list = frames[1]
		#plot
		#plot ground truth
		ax = plt.Subplot(fig, inner[0])
		x_lim = image_data_gt.shape[1]
		#plot only a portion of gt for comparison
		index_current = stride_step*(window_size+i+(window_size_predicted-overlap)*repeat_prediction)
		ax.imshow(image_data_gt[:,:index_current,:])
		ax.set_xlim(0, x_lim)
		if labels_dict:
			defect_label_gt = labels_dict['gt'][i]
			defect_type_gt = get_defect_type(defect_label_gt)
		else:
			defect_type_gt = 'No label'
		if defect_type_gt=='Puck':
			defect_type_gt='Pucker'
		ax.set_title(f'Ground truth. {defect_type_gt}', fontweight='bold',fontsize=fontsize_sec+2)
		#plot separator input/prediction
		x_separator = stride_step*(window_size+i)
		ax.axvline(x = x_separator, color = 'red',linewidth=linewidth)
		#add window borders
		x_window_left = x_separator-stride_step*window_size
		ax.axvline(x = x_window_left, color = 'orange',linewidth=linewidth)
		x_window_right = x_separator+stride_step*((window_size_predicted-overlap)*repeat_prediction)
		ax.axvline(x = x_window_right, color = 'orange',linewidth=linewidth)
		ax.set_yticks([])
		fig.add_subplot(ax)

		for ii, frames_prediction in enumerate(frames_prediction_list):
			unit_numb = unit_numb_list[ii]
			if labels_dict:
				try:
					defect_label_predicted = labels_dict[unit_numb][0][i]
					defect_type_predicted = get_defect_type(defect_label_predicted)
				except KeyError:
					defect_type_predicted = ''
			else:
				defect_type_predicted = 'No label'
			image_data_predicted = assemble_image(frames_prediction)
			#plot prediction
			ax = plt.Subplot(fig, inner[ii+1])
			#reassemble to include all previous gt
			image_data_plot = np.concatenate(
				(
					image_data_gt[:,:x_separator,:],
					image_data_predicted
				),
				axis=1
			)
			ax.imshow(image_data_plot)
			if defect_type_predicted=='Puck':
				defect_type_predicted='Pucker'
			ax.set_title(f'Prediction for LS = {unit_numb}. {defect_type_predicted}',fontsize=fontsize_sec+2, fontweight='bold')
			ax.set_xlim(0, x_lim)
			ax.axvline(x = x_separator, color = 'red',linewidth=linewidth)
			ax.axvline(x = x_window_left, color = 'orange',linewidth=linewidth)
			ax.axvline(x = x_window_right, color = 'orange',linewidth=linewidth)
			ax.set_yticks([])
			fig.add_subplot(ax)

		
		if ifMetrics:

			inner_error = gridspec.GridSpecFromSubplotSpec(
						1, 2,
						subplot_spec=inner[-1], wspace=0.25
						)


			#plot errors
			ssim_frames_list_temp = []
			mse_gray_frames_list_temp = []
			mse_rgb_frames_list_temp = []
			if 1:
				ax_mse_rgb = plt.Subplot(fig, inner_error[0])
				#ax_mse_gray = plt.Subplot(fig, inner_error[1])
				ax_ssim = plt.Subplot(fig, inner_error[1])
			#ax_mse_rgb.set_title(f'MSE RGB')
			ax_mse_rgb.set_title(f'MSE', fontweight='bold',fontsize=fontsize_sec)
			#ax_mse_gray.set_title(f'MSE Gray') 
			ax_ssim.set_title(f'SSIM', fontweight='bold',fontsize=fontsize_sec)  

			for ii, frames_prediction in enumerate(frames_prediction_list):

				ssim_frames, mse_gray_frames, mse_rgb_frames = get_errors(
					frames_gt, 
					frames_prediction
						)
				ssim_frames_list_temp.append(ssim_frames)
				mse_gray_frames_list_temp.append(mse_gray_frames)
				mse_rgb_frames_list_temp.append(mse_rgb_frames)
				
				#ax_mse_gray.plot(mse_gray_frames, label=f'ls: {unit_numb_list[ii]}')
				ax_mse_rgb.plot(mse_rgb_frames, label=f'ls: {unit_numb_list[ii]}')			
				ax_ssim.plot(ssim_frames, label=f'ls: {unit_numb_list[ii]}')

			#ax_mse_rgb.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			ax_mse_rgb.legend(loc='upper right',fontsize=fontsize_sec-2)
			#ax_mse_gray.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			#ax_ssim.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
			ax_ssim.legend(loc='upper right',fontsize=fontsize_sec-2)
			extra_length = 3 #to fit the legend without overlapping with graphs
			x_lim = len(mse_gray_frames)+extra_length
			ax_mse_rgb.set_xlim(0, x_lim)
			ax_mse_rgb.set_xlabel('predicted frame number', fontweight='bold',fontsize=fontsize_sec)
			ax_mse_rgb.set_xticks(
				np.arange(0, repeat_prediction, step=1), 
				labels=[x+1 for x in range(repeat_prediction)],
				#fontsize=fontsize_sec-2,
				)
			ax_mse_rgb.tick_params(axis='both', labelsize=fontsize_sec-3)
			#ax_mse_gray.set_xlim(0, x_lim)
			ax_ssim.set_xlim(0, x_lim)
			ax_ssim.set_xlabel('predicted frame number', fontweight='bold',fontsize=fontsize_sec)
			ax_ssim.set_xticks(
				np.arange(0, repeat_prediction, step=1),
				labels=[x+1 for x in range(repeat_prediction)],
				#fontsize=fontsize_sec-2,
				)
			ax_ssim.tick_params(axis='both', labelsize=fontsize_sec-3)

			if 1:
				fig.add_subplot(ax_mse_rgb)
				#fig.add_subplot(ax_mse_gray)
				fig.add_subplot(ax_ssim)

				
			ssim_frames_list.append(ssim_frames_list_temp)
			mse_gray_frames_list.append(mse_gray_frames_list_temp)
			mse_rgb_frames_list.append(mse_rgb_frames_list_temp)

		if ifSave:
			filename = f'{save_folder}/step_{i+1}.svg'
			plt.savefig(filename, bbox_inches='tight')

		if ifClose:
			plt.close()


