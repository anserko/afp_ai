import tensorflow as tf
from tensorflow.keras import layers, models

from build_model_ed import build_model_2, build_model_3

def get_encoder_decoder(model_name, model_weights_path, model_settings, loss, optimizer, ifEncoder=True, ifDecoder=True, ifSummary=True):


	input_shape, dense_units,ifBatchNorm,kernel_size,nn_blocks = model_settings


	models_list = []
	#main model	
	#model = build_model_2(input_shape, dense_units, nn_blocks)
	model = build_model_3(input_shape, dense_units, nn_blocks)
	model.compile(loss=loss, 
	              optimizer=optimizer, )
	model.load_weights(model_weights_path).expect_partial()

	if ifEncoder:
		#encoder model
		layers_encoder = model.layers[:len(model.layers)//2]
		model_encoder = models.Sequential()
		for layer in layers_encoder:
		    model_encoder.add(layer)

		model_encoder.compile(loss=loss, 
		              optimizer=optimizer)
		models_list.append(model_encoder)
	else:
		models_list.append(None)

	if ifDecoder:
		#decoder
		layers_decoder = model.layers[len(model.layers)//2:]
		model_decoder = models.Sequential()
		input_layer_decoder_shape = layers_decoder[0].input_shape[-1]
		model_decoder.add(layers.Input(shape=input_layer_decoder_shape))
		for layer in layers_decoder:
		    model_decoder.add(layer)

		model_decoder.compile(loss=loss, 
		              optimizer=optimizer)
		models_list.append(model_decoder)
	else:
		models_list.append(None)


	if ifSummary:
		print('\n#####Main model model#####\n')
		model.summary() 
		if ifEncoder:
			print('\n#####Encoder model#####\n')
			model_encoder.summary()
		if ifDecoder:
			print('\n#####Decoder model#####\n')
			model_decoder.summary() 
		print('\n')
		if ifEncoder:
			print(f'Encoder input shape: {model_encoder.layers[0].input_shape}')
			print(f'Encoder output shape: {model_encoder.layers[-1].output_shape}')
		if ifDecoder:
			print(f'Decoder input shape: {model_decoder.layers[0].input_shape}')
			print(f'Decoder output shape: {model_decoder.layers[-1].output_shape}')

	return models_list
