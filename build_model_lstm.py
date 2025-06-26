import tensorflow as tf
from tensorflow.keras import layers, models


def build_model_lstm_1(input_shape, cells_list, ifDense, ifDropout=False):
    """
    Create a neural network model, get current force, output next step thickness

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, channels)

    cells_list : list
        list of LSTM layers, each element represent a number of lstm cells

    ifDense : boolean
        if True output layer is Dense, if False output layer is Conv1D

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    # input
    input_layer = layers.Input(shape=input_shape)

    last_layer = input_layer
    for cells in cells_list:
        # lstm_layer = layers.LSTM(cells, recurrent_activation='sigmoid', return_sequences=True)(last_layer)
        lstm_layer = layers.LSTM(cells, return_sequences=True)(last_layer)
        #lstm_layer = layers.LSTM(cells, return_sequences=True, activation='relu')(last_layer)
        last_layer = lstm_layer

    if ifDropout:
        dropout_layer = layers.Dropout(0.2)(last_layer)
        last_layer = dropout_layer

    if ifDense:
        units_dense = input_shape[-1]
        output_layer = layers.Dense(units=units_dense, activation="sigmoid")(last_layer)
    else:
        filters = input_shape[-1]
        output_layer = layers.Conv1D(filters=filters,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(last_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_lstm_2(input_shape, cells_list, ifDense):
    """
    Create a neural network model, get current force, output next step thickness

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, channels)

    cells_list : list
        list of LSTM layers, each element represent a number of lstm cells

    ifDense : boolean
        if True output layer is Dense, if False output layer is Conv1D

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    #input
    input_layer = layers.Input(shape=input_shape)

    #intermed_layer = layers.Conv1D(filters=16,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(input_layer)
    intermed_layer = layers.Dense(units=150, activation="sigmoid")(input_layer)

    last_layer = intermed_layer
    for cells in cells_list:
        #lstm_layer = layers.LSTM(cells, recurrent_activation='sigmoid', return_sequences=True)(last_layer)
        lstm_layer = layers.LSTM(cells, return_sequences=True)(last_layer)
        last_layer = lstm_layer

    if ifDense:
        units_dense = input_shape[-1]
        output_layer = layers.Dense(units=units_dense, activation="sigmoid")(last_layer)
    else:
        filters = input_shape[-1]
        output_layer = layers.Conv1D(filters=filters,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(last_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_lstm_3(input_shape, cells_list, ifDense, ifDropout=False, many_to_many=False):
    """
    Create a neural network model, get current force, output next step thickness

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, channels)

    cells_list : list
        list of LSTM layers, each element represent a number of lstm cells

    ifDense : boolean
        if True output layer is Dense, if False output layer is Conv1D

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    # input
    input_layer = layers.Input(shape=input_shape)

    last_layer = input_layer
    return_sequences=True
    for i, cells in enumerate(cells_list):
        # lstm_layer = layers.LSTM(cells, recurrent_activation='sigmoid', return_sequences=True)(last_layer)
        if i==(len(cells_list)-1) and not many_to_many:
            return_sequences=False
        lstm_layer = layers.LSTM(cells, return_sequences=return_sequences)(last_layer)
        #lstm_layer = layers.LSTM(cells, return_sequences=True, activation='relu')(last_layer)
        last_layer = lstm_layer

    if ifDropout:
        dropout_layer = layers.Dropout(0.2)(last_layer)
        last_layer = dropout_layer

    if ifDense:
        units_dense = input_shape[-1]
        output_layer = layers.Dense(units=units_dense, activation="sigmoid")(last_layer)
    else:
        filters = input_shape[-1]
        output_layer = layers.Conv1D(filters=filters,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(last_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model
