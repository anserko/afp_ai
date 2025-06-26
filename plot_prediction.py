import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from get_errors import get_ssim, get_mse, get_errors
from assemble_image import assemble_image
from get_labels import get_defect_type
from load_image import load_image

def plot_prediction(image_data_frames_gt,predicted_data_dict,labels_dict,lstm_pars_dict,plot_settings_dict):


	ifSingle = plot_settings_dict['ifSingle']
	if ifSingle:
		plot_prediction_single(image_data_frames_gt,predicted_data_dict,labels_dict,lstm_pars_dict,plot_settings_dict)
	else:
		plot_prediction_all(image_data_frames_gt,predicted_data_dict,labels_dict,lstm_pars_dict,plot_settings_dict)
	


def plot_prediction_single(image_data_frames_gt,predicted_data_dict,labels_dict,lstm_pars_dict,plot_settings_dict):
	
	#exclude MSE Gray metric
	ifMetrics = plot_settings_dict['ifMetrics']
	ifSave = plot_settings_dict['ifSave']
	save_folder = plot_settings_dict['save_folder']
	ifClose = plot_settings_dict['ifClose']

	unit_numb_list = list(predicted_data_dict.keys())
	predicted_data_list = [predicted_data_dict[x] for x in unit_numb_list]

	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	repeat_prediction = lstm_pars_dict['repeat_prediction']
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



def plot_prediction_all(image_data_frames_gt,predicted_data_dict,labels_dict,lstm_pars_dict,plot_settings_dict):
	
	numb_cols = plot_settings_dict['numb_cols']
	ifMetrics = plot_settings_dict['ifMetrics']
	ifSave = plot_settings_dict['ifSave']
	save_folder = plot_settings_dict['save_folder']
	ifClose = plot_settings_dict['ifClose']

	unit_numb_list = list(predicted_data_dict.keys())
	predicted_data_list = [predicted_data_dict[x] for x in unit_numb_list]

	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	repeat_prediction = lstm_pars_dict['repeat_prediction']
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

	if ifSave:
		filename = f'{save_folder}/all.svg'
		plt.savefig(filename, bbox_inches='tight')

	if ifClose:
			plt.close()


def plot_probabilities(labels_dict, lstm_pars_dict, stride_step, image_name, plot_settings_dict):

	ifSave = plot_settings_dict['ifSave']
	save_folder = plot_settings_dict['save_folder']

	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	repeat_prediction = lstm_pars_dict['repeat_prediction']
	window_total = window_size + (window_size_predicted-overlap)*repeat_prediction
	#window_total = window_total//2

	image_data = load_image(image_name)
	image_length = image_data.shape[1]

	fig, axes = plt.subplots(2+(len(labels_dict)-1),1, figsize=(20,20+10*(len(labels_dict)-1)))
	fig.subplots_adjust(hspace=0.4, wspace=0.1)
	linewidth = 3.0

	fontsize_sec = 25
	fontsize_title = fontsize_sec+12
	fontsize_label = fontsize_sec+10
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
	ax.plot(x, labels_gt, linewidth=linewidth, marker=marker, markersize=markersize, label='Real AFP sensor')
	for key in labels_dict.keys():
		if type(key)!=int:
			#skip gt key
			continue
		labels_predicted = labels_dict[key][0]
		ax.plot(x, labels_predicted, linewidth=linewidth, marker=marker, markersize=markersize, label=f'Predictive sensor. LS: {key}')
	ax.legend(fontsize=fontsize_legend, loc='upper left')
	ax.set_title(f'Real vs Predictive sensor defect detection', fontsize=fontsize_title, fontweight='bold') 
	ax.set_xlabel('Sensor location', fontsize=fontsize_label, fontweight='bold')
	ax.set_ylabel('Defect type label', fontsize=fontsize_label, fontweight='bold')	
	ax.set_xlim(0, image_length)

	labels = [get_defect_type(x) if x!='No defect' else 'No\ndefect' for x in range(4)]
	ax.set_yticks(
				np.arange(0, 4, step=1),
				labels=[x if x!='No defect' else 'No\ndefect' for x in labels],
				)
	ax.tick_params(axis='both', labelsize=fontsize_label)

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
		ax.set_title(f'Predictive sensor probabilities. LS: {key}', fontsize=fontsize_title, fontweight='bold') 
		ax.set_xlabel('Sensor location', fontsize=fontsize_label, fontweight='bold')
		ax.set_ylabel('Defect type probability', fontsize=fontsize_label, fontweight='bold')
		ax.tick_params(axis='both', labelsize=fontsize_label)
		ax.set_xlim(0, image_length)

	if ifSave:
		filename = f'{save_folder}/probabilities.svg'
		plt.savefig(filename, bbox_inches='tight')