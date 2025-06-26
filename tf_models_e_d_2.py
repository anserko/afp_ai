import tensorflow as tf
from tensorflow.keras import layers, models



def build_model_encoded_try(input_shape, cells_list, ifBatchNorm, ifDense):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, channels)

    cells_list : list
        list of LSTM layers, each element represent a number of lstm cells

    ifBatchNorm : boolean
        include BatchNormalisation layer after each LSTM layer or not

    ifDense : boolean
        if True output layer is Dense, if False output layer is Conv1D

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    #input
    input_layer = layers.Input(shape=input_shape)

    last_layer = input_layer
    for cells in cells_list:
        lstm_layer = layers.LSTM(cells, return_sequences=True)(last_layer)
        last_layer = lstm_layer
        if ifBatchNorm:
            bn_layer = layers.BatchNormalization()(last_layer)
            last_layer = bn_layer

    if ifDense:
        units_dense = input_shape[-1]
        output_layer = layers.Dense(units=units_dense, activation="sigmoid")(last_layer)
    else:
        filters = input_shape[-1]
        output_layer = layers.Conv1D(filters=filters,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(last_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

