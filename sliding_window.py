import numpy as np


def sliding_window(data, window_size, window_size_predicted, overlap):
	#processing the whole dataset
	#input data
	x_train_lstm = np.lib.stride_tricks.sliding_window_view(
		data,
		window_shape = window_size,
		axis=1,
		)
	#cut off last timesteps
	x_train_lstm = x_train_lstm[:,:-(window_size_predicted-overlap),:,:]
	#transpose
	x_train_lstm = np.transpose(x_train_lstm, axes=(0,1,3,2))
	shape = x_train_lstm.shape
	x_train_lstm = np.reshape(x_train_lstm,(shape[0]*shape[1], shape[2], shape[3]))

	#labels
	y_train_lstm = np.lib.stride_tricks.sliding_window_view(
	    data,
	    window_shape = window_size_predicted,
	    axis=1,
	)
	#shift
	y_train_lstm = y_train_lstm[:,(window_size-overlap):,:,:]
	#transpose
	y_train_lstm = np.transpose(y_train_lstm, axes=(0,1,3,2))
	#join batches from different timeseries together
	shape = y_train_lstm.shape
	y_train_lstm = np.reshape(y_train_lstm,(shape[0]*shape[1], shape[2], shape[3]))

	if window_size_predicted==1:
		#if only one frame is predicted
		#get rid of timestep dim
		shape = y_train_lstm.shape
		y_train_lstm = np.reshape(y_train_lstm,(shape[0], shape[1]*shape[2]))

	return x_train_lstm, y_train_lstm


