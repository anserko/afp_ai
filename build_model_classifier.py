import tensorflow as tf
from tensorflow.keras import layers, models


def build_model_classifier_1(input_shape):

	dropout = 0.1
	input_layer = layers.Input(shape=input_shape)

	flatten_layer = layers.Flatten()(input_layer)
	dense_layer = layers.Dense(100, activation='relu')(flatten_layer)
	dense_layer = layers.Dropout(dropout)(dense_layer)
	dense_layer = layers.Dense(100, activation='relu')(dense_layer)
	dense_layer = layers.Dropout(dropout)(dense_layer)
	dense_layer = layers.Dense(100, activation='relu')(dense_layer)

	output_layer = layers.Dense(4)(dense_layer)

	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

	return model

def build_model_classifier_2(input_shape):

	input_layer = layers.Input(shape=input_shape)

	conv_layer = layers.Conv1D(4, 3, padding='same', activation='relu')(input_layer)
	pool_layer = layers.MaxPooling1D()(conv_layer)
	conv_layer = layers.Conv1D(8, 3, padding='same', activation='relu')(pool_layer)
	pool_layer = layers.MaxPooling1D()(conv_layer)

	flatten_layer = layers.Flatten()(pool_layer)

	dense_layer = layers.Dense(100, activation='relu')(flatten_layer)
	dense_layer = layers.Dense(100, activation='relu')(dense_layer)

	output_layer = layers.Dense(4)(dense_layer)

	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

	return model

def build_model_classifier_3(settings_dict):

	input_shape = settings_dict['input_shape']
	conv_blocks_list = settings_dict['conv_blocks_list']
	dense_blocs_list = settings_dict['dense_blocs_list']
	labels_nmb = settings_dict['labels_nmb']

	input_layer = layers.Input(shape=input_shape)

	last_layer = input_layer
	for conv_block in conv_blocks_list:
		conv_layer = layers.Conv1D(conv_block, 3, padding='same', activation='relu')(last_layer)
		pool_layer = layers.MaxPooling1D()(conv_layer)
		last_layer = pool_layer

	flatten_layer = layers.Flatten()(last_layer)
	last_layer = flatten_layer
	for dense_block in dense_blocs_list:
		dense_layer = layers.Dense(dense_block, activation='relu')(last_layer)
		last_layer = dense_layer

	output_layer = layers.Dense(labels_nmb)(last_layer)

	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

	return model
