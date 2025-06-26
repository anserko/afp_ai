import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json

from build_model_lstm import build_model_lstm_3
from build_model_classifier import build_model_classifier_3
from get_encoder_decoder import get_encoder_decoder

def load_encoder_decoder_model(model_ed_name, model_ed_path):
	with open(f'{model_ed_path}/{model_ed_name}_settings.json', 'r') as f:
		ed_settings_dict = json.load(f)

	input_shape = ed_settings_dict['input_shape']
	dense_units = ed_settings_dict['dense_units']
	ifBatchNorm = ed_settings_dict['ifBatchNorm']
	kernel_size = ed_settings_dict['kernel_size']
	nn_blocks = ed_settings_dict['nn_blocks']

	ifEncoder = True
	ifDecoder = True
	ifSummary = False
	loss = 'mean_squared_error'
	optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=0.1, name="Adadelta")   

	model_settings = (input_shape, dense_units,ifBatchNorm,kernel_size,nn_blocks)
	model_weights_path = f'{model_ed_path}/{model_ed_name}/{model_ed_name}'    

	model_encoder, model_decoder = get_encoder_decoder(
		model_ed_name,
		model_weights_path,
		model_settings,
		loss, optimizer, 
		ifEncoder=ifEncoder,
		ifDecoder=ifDecoder,
		ifSummary=ifSummary
		)

	return model_encoder, model_decoder

def load_lstm_model(model_lstm_name, model_lstm_path):

	with open(f'{model_lstm_path}/{model_lstm_name}_settings.json', 'r') as f:
		lstm_settings_dict_test = json.load(f)

	unit_numb = lstm_settings_dict_test['unit_numb']
	cells_list = lstm_settings_dict_test['cells_list']
	ifDense = lstm_settings_dict_test['ifDense']
	many_to_many = lstm_settings_dict_test['many_to_many']
	stateful = lstm_settings_dict_test['stateful']

	input_shape = (None, unit_numb)

	model = build_model_lstm_3(
		input_shape,
		cells_list,
		ifDense=ifDense,
		ifDropout=False,
		many_to_many=many_to_many
		)
	model.stateful = stateful

	loss = 'mean_squared_error'
	optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=0.1, name="Adadelta")
	model.compile(loss=loss, optimizer=optimizer)

	model.load_weights(f'{model_lstm_path}/{model_lstm_name}/{model_lstm_name}').expect_partial()  

	return model

def load_classification_model(model_name, model_path, ifProbability):

	with open(f'{model_path}/{model_name}_settings.json', 'r') as f:
		classification_settings_dict_test = json.load(f)

	optimizer = 'adam'
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	#model = model = build_model_classifier_3(classification_settings_dict_test)
	model = build_model_classifier_3(classification_settings_dict_test)
	model.compile(
		loss=loss,
		optimizer=optimizer,
		metrics=['accuracy']
	)

	model.load_weights(f'{model_path}/{model_name}/{model_name}').expect_partial()  

	if ifProbability:
		output_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
	else:
		output_model = model
	return output_model

def load_models(unit_numb, stride_step, lstm_pars_dict, classifier=False, ifProbability=True):

	models_dict={}

	#load encoder decoder models
	model_ed_name = f'model_ed_2_ps_2_bn_True_du_{unit_numb}'
	model_ed_path = f'saved_models/stride_{stride_step}/ed'
	model_encoder, model_decoder = load_encoder_decoder_model(model_ed_name, model_ed_path)

	#load lstm models
	stateful = lstm_pars_dict['stateful']
	window_size = lstm_pars_dict['window_size']
	window_size_predicted = lstm_pars_dict['window_size_predicted']
	overlap = lstm_pars_dict['overlap']
	lstm_type = lstm_pars_dict['lstm_type']

	if window_size_predicted==1:
		if lstm_type == 'large_quasi_state_win':
			cells_list = [unit_numb*4]
		else:
			cells_list = [unit_numb*2]
	else:
		if lstm_type == 'large_quasi_state_win':
			cells_list = [unit_numb*4,unit_numb*4,unit_numb*4]
		else:
			cells_list = [unit_numb*2,unit_numb*2]
	cells_list_str = '_'.join(str(x) for x in cells_list)

	model_lstm_name = f'model_lstm_{lstm_type}_{window_size}_{window_size_predicted}_ovrp_{overlap}_un_{unit_numb}_c_{cells_list_str}'
	#model_lstm_name = f'model_lstm_large_quasi_state_win_{window_size}_{window_size_predicted}_ovrp_{overlap}_un_{unit_numb}_c_{cells_list_str}'  
	#model_lstm_name = f'model_lstm_quasi_state_win_{window_size}_{window_size_predicted}_ovrp_{overlap}_un_{unit_numb}_c_{cells_list_str}'             
	model_lstm_path = f'saved_models/stride_{stride_step}/lstm/lstm_state_{stateful}'
	model_lstm = load_lstm_model(model_lstm_name, model_lstm_path)
	

	models_dict['model_encoder'] = model_encoder
	models_dict['model_decoder'] = model_decoder
	models_dict['model_lstm'] = model_lstm

	if classifier:
		model_classifier_name = f'model_class_str_{stride_step}_un_{unit_numb}'
		model_classifier_path = f'saved_models/stride_{stride_step}/classify'
		model_classifier = load_classification_model(model_classifier_name, model_classifier_path, ifProbability=ifProbability)
		models_dict['model_classifier'] = model_classifier

	return models_dict


