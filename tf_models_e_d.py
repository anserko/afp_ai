import tensorflow as tf
from tensorflow.keras import layers, models


def build_model_e_d(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, nx, ny, nz, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input
    input_layer = layers.Input(shape=input_shape)
    #zero padding is addedd to avoid shape mismatch after upsampling due to odd numbers
    zeropad = layers.ZeroPadding3D(padding=(0, 1, 0))(input_layer)

    #encoder
    conv1 = layers.Conv3D(filters=32,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(zeropad)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=64,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pooling1 = layers.MaxPooling3D(pool_size=(2,2,2))(conv1)


    conv2 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pooling2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)


    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=800,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    print(f'Last conv layer shape: {conv3.shape}')

    #reshape = layers.Reshape((-1, input_shape[-1]*nx*ny*nz))(conv3)
    flatten = layers.Flatten()(conv3)
    dense_units = 400
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(conv3.shape[1:])(dense2)


    #decoder
    conv4 = layers.Conv3D(filters=800,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(reshape1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    upsampling1 = layers.UpSampling3D((2,2,2))(conv4)

    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling1)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    upsampling2 = layers.UpSampling3D((2,2,2))(conv5)

    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=64,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=32,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=3,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    #crop last layer to take into acccount zero padding
    decoded_cropping = layers.Cropping3D((0,1,0))(conv6)

    #output_layer = layers.Conv3D(filters=64,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    output_layer = decoded_cropping


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_encoder(input_shape):
    """
    Create neural network model for encoder

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, nx, ny, nz, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input
    input_layer_encoder = layers.Input(shape=input_shape)
    #zero padding is addedd to avoid shape mismatch after upsampling due to odd numbers
    zeropad = layers.ZeroPadding3D(padding=(0, 1, 0))(input_layer_encoder)

    #encoder
    conv1 = layers.Conv3D(filters=32,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(zeropad)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=64,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pooling1 = layers.MaxPooling3D(pool_size=(2,2,2))(conv1)


    conv2 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pooling2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)


    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=800,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    print(f'Last conv layer shape: {conv3.shape}')

    #reshape = layers.Reshape((-1, input_shape[-1]*nx*ny*nz))(conv3)
    flatten = layers.Flatten()(conv3)
    dense_units = 400
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    
    model_encoder = tf.keras.Model(inputs=input_layer_encoder, outputs=dense1)

    return model_encoder, dense1.shape, flatten.shape[-1], conv3.shape


def build_model_decoder(input_shape, dense_units, conv_shape):
    """
    Create neural network model decoder

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, nx, ny, nz, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input
    input_layer_decoder = layers.Input(shape=input_shape)
    
    dense2 = layers.Dense(units=dense_units, activation="sigmoid")(input_layer_decoder)
    reshape1 = layers.Reshape(conv_shape[1:])(dense2)


    #decoder
    conv4 = layers.Conv3D(filters=800,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(reshape1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    upsampling1 = layers.UpSampling3D((2,2,2))(conv4)

    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling1)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    upsampling2 = layers.UpSampling3D((2,2,2))(conv5)

    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=64,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=32,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=3,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    #crop last layer to take into acccount zero padding
    decoded_cropping = layers.Cropping3D((0,1,0))(conv6)

    model_decoder = tf.keras.Model(inputs=input_layer_decoder, outputs=decoded_cropping)

    return model_decoder


def build_model_e_d_2(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, nx, ny, nz, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input
    input_layer = layers.Input(shape=input_shape)
    #zero padding is addedd to avoid shape mismatch after upsampling due to odd numbers
    zeropad = layers.ZeroPadding3D(padding=(0, 1, 0))(input_layer)

    #encoder
    conv1 = layers.Conv3D(filters=32,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(zeropad)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=64,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pooling1 = layers.MaxPooling3D(pool_size=(2,2,2))(conv1)


    conv2 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pooling2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)


    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=1024,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    print(f'Last conv layer shape: {conv3.shape}')

    #reshape = layers.Reshape((-1, input_shape[-1]*nx*ny*nz))(conv3)
    flatten = layers.Flatten()(conv3)
    dense_units = 500
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(conv3.shape[1:])(dense2)


    #decoder
    conv4 = layers.Conv3D(filters=1024,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(reshape1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    upsampling1 = layers.UpSampling3D((2,2,2))(conv4)

    conv5 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling1)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    upsampling2 = layers.UpSampling3D((2,2,2))(conv5)

    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=64,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=32,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=3,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    #crop last layer to take into acccount zero padding
    decoded_cropping = layers.Cropping3D((0,1,0))(conv6)

    #output_layer = layers.Conv3D(filters=64,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    output_layer = decoded_cropping


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_e_d_3(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, nx, ny, nz, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input
    input_layer = layers.Input(shape=input_shape)
    #zero padding is addedd to avoid shape mismatch after upsampling due to odd numbers
    zeropad = layers.ZeroPadding3D(padding=(0, 1, 0))(input_layer)

    #encoder
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(zeropad)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pooling1 = layers.MaxPooling3D(pool_size=(2,2,2))(conv1)


    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pooling2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)


    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    print(f'Last conv layer shape: {conv3.shape}')

    #reshape = layers.Reshape((-1, input_shape[-1]*nx*ny*nz))(conv3)
    flatten = layers.Flatten()(conv3)
    dense_units = 500
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(conv3.shape[1:])(dense2)


    #decoder
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(reshape1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    upsampling1 = layers.UpSampling3D((2,2,2))(conv4)

    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling1)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    upsampling2 = layers.UpSampling3D((2,2,2))(conv5)

    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    #crop last layer to take into acccount zero padding
    cropping = layers.Cropping3D((0,1,0))(conv6)

    output = layers.Conv3D(filters=3,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(cropping)

    output_layer = output


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_e_d_4(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, nx, ny, nz, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input
    input_layer = layers.Input(shape=input_shape)
    #zero padding is addedd to avoid shape mismatch after upsampling due to odd numbers
    zeropad = layers.ZeroPadding3D(padding=(0, 1, 0))(input_layer)

    #encoder
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(zeropad)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pooling1 = layers.MaxPooling3D(pool_size=(2,2,2))(conv1)


    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pooling2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)


    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=700,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    print(f'Last conv layer shape: {conv3.shape}')

    #reshape = layers.Reshape((-1, input_shape[-1]*nx*ny*nz))(conv3)
    flatten = layers.Flatten()(conv3)
    dense_units = 500
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(conv3.shape[1:])(dense2)


    #decoder
    conv4 = layers.Conv3D(filters=700,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(reshape1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    upsampling1 = layers.UpSampling3D((2,2,2))(conv4)

    conv5 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling1)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    upsampling2 = layers.UpSampling3D((2,2,2))(conv5)

    conv6 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    #crop last layer to take into acccount zero padding
    cropping = layers.Cropping3D((0,1,0))(conv6)

    output = layers.Conv3D(filters=3,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(cropping)

    output_layer = output


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def build_model_e_d_4_reduced(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, nx, ny, nz, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input
    input_layer = layers.Input(shape=input_shape)
    #zero padding is addedd to avoid shape mismatch after upsampling due to odd numbers
    zeropad = layers.ZeroPadding3D(padding=(0, 1, 0))(input_layer)

    #encoder
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(zeropad)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pooling1 = layers.MaxPooling3D(pool_size=(2,2,2))(conv1)


    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pooling2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)


    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=700,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    print(f'Last conv layer shape: {conv3.shape}')

    #reshape = layers.Reshape((-1, input_shape[-1]*nx*ny*nz))(conv3)
    flatten = layers.Flatten()(conv3)
    dense_units = 100
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(conv3.shape[1:])(dense2)


    #decoder
    conv4 = layers.Conv3D(filters=700,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(reshape1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    upsampling1 = layers.UpSampling3D((2,2,2))(conv4)

    conv5 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling1)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    upsampling2 = layers.UpSampling3D((2,2,2))(conv5)

    conv6 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    #crop last layer to take into acccount zero padding
    cropping = layers.Cropping3D((0,1,0))(conv6)

    output = layers.Conv3D(filters=3,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(cropping)

    output_layer = output


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_e_d_4_reduced_250(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, nx, ny, nz, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input
    input_layer = layers.Input(shape=input_shape)
    #zero padding is addedd to avoid shape mismatch after upsampling due to odd numbers
    zeropad = layers.ZeroPadding3D(padding=(0, 1, 0))(input_layer)

    #encoder
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(zeropad)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pooling1 = layers.MaxPooling3D(pool_size=(2,2,2))(conv1)


    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pooling2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)


    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(pooling2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(filters=700,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    print(f'Last conv layer shape: {conv3.shape}')

    #reshape = layers.Reshape((-1, input_shape[-1]*nx*ny*nz))(conv3)
    flatten = layers.Flatten()(conv3)
    dense_units = 250
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(conv3.shape[1:])(dense2)


    #decoder
    conv4 = layers.Conv3D(filters=700,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(reshape1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(filters=512,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    upsampling1 = layers.UpSampling3D((2,2,2))(conv4)

    conv5 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling1)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=400,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    upsampling2 = layers.UpSampling3D((2,2,2))(conv5)

    conv6 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(upsampling2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=256,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv3D(filters=128,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    #crop last layer to take into acccount zero padding
    cropping = layers.Cropping3D((0,1,0))(conv6)

    output = layers.Conv3D(filters=3,kernel_size=(3, 3, 3),activation="sigmoid",padding="same",data_format='channels_last')(cropping)

    output_layer = output


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model




def build_model_encoded_1(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    #input
    input_layer = layers.Input(shape=input_shape)

    conv1 = layers.Conv1D(filters=16,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(input_layer)
    lstm1 = layers.LSTM(16, return_sequences=True)(conv1)
    bn1 = layers.BatchNormalization()(lstm1)

    conv2 = layers.Conv1D(filters=32,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn1)
    lstm2 = layers.LSTM(32, return_sequences=True)(conv2)
    bn2 = layers.BatchNormalization()(lstm2)

    conv3 = layers.Conv1D(filters=64,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn2)
    lstm3 = layers.LSTM(64, return_sequences=True)(conv3)
    bn3 = layers.BatchNormalization()(lstm3)

    conv4 = layers.Conv1D(filters=128,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn3)
    lstm4 = layers.LSTM(128, return_sequences=True)(conv4)
    bn4 = layers.BatchNormalization()(lstm4)

    conv5 = layers.Conv1D(filters=256,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn4)
    lstm5 = layers.LSTM(256, return_sequences=True)(conv5)
    bn5 = layers.BatchNormalization()(lstm5)

    conv6 = layers.Conv1D(filters=512,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn5)
    lstm6 = layers.LSTM(512, return_sequences=True)(conv6)
    bn6 = layers.BatchNormalization()(lstm6)

    conv7 = layers.Conv1D(filters=1024,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn6)
    lstm7 = layers.LSTM(1024, return_sequences=True)(conv7)
    bn7 = layers.BatchNormalization()(lstm7)

    conv8 = layers.Conv1D(filters=2048,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn7)
    lstm8 = layers.LSTM(2048, return_sequences=True)(conv8)
    bn8 = layers.BatchNormalization()(lstm8)

    dense_units = 500
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(bn8)
    

    output_layer = dense1


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model




def build_model_encoded_2(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    #input
    input_layer = layers.Input(shape=input_shape)

    #conv1 = layers.Conv1D(filters=16,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(input_layer)
    lstm1 = layers.LSTM(16, return_sequences=True)(input_layer)
    bn1 = layers.BatchNormalization()(lstm1)

    #conv2 = layers.Conv1D(filters=32,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn1)
    lstm2 = layers.LSTM(32, return_sequences=True)(bn1)
    bn2 = layers.BatchNormalization()(lstm2)

    #conv3 = layers.Conv1D(filters=64,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn2)
    lstm3 = layers.LSTM(64, return_sequences=True)(bn2)
    bn3 = layers.BatchNormalization()(lstm3)

    #conv4 = layers.Conv1D(filters=128,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn3)
    lstm4 = layers.LSTM(128, return_sequences=True)(bn3)
    bn4 = layers.BatchNormalization()(lstm4)

    #conv5 = layers.Conv1D(filters=256,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn4)
    lstm5 = layers.LSTM(256, return_sequences=True)(bn4)
    bn5 = layers.BatchNormalization()(lstm5)

    #conv6 = layers.Conv1D(filters=512,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn5)
    lstm6 = layers.LSTM(512, return_sequences=True)(bn5)
    bn6 = layers.BatchNormalization()(lstm6)

    #conv7 = layers.Conv1D(filters=1024,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn6)
    lstm7 = layers.LSTM(1024, return_sequences=True)(bn6)
    bn7 = layers.BatchNormalization()(lstm7)

    #conv8 = layers.Conv1D(filters=2048,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn7)
    lstm8 = layers.LSTM(2048, return_sequences=True)(bn7)
    bn8 = layers.BatchNormalization()(lstm8)

    filters = 500
    conv1 = layers.Conv1D(filters=filters,kernel_size=3,activation="sigmoid",padding="same",data_format='channels_last')(bn8)
    

    output_layer = conv1


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def build_model_encoded_3(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    #input
    input_layer = layers.Input(shape=input_shape)

    units1 = 16
    lstm1 = layers.LSTM(units=units1, return_sequences=True)(input_layer)
    dense1 = layers.Dense(units=units1, activation="sigmoid")(lstm1)
    bn1 = layers.BatchNormalization()(dense1)

    units2 = 32
    lstm2 = layers.LSTM(units=units2, return_sequences=True)(bn1)
    dense2 = layers.Dense(units=units2, activation="sigmoid")(lstm2)
    bn2 = layers.BatchNormalization()(dense2)

    units3 = 64
    lstm3 = layers.LSTM(units=units3, return_sequences=True)(bn2)
    dense3 = layers.Dense(units=units3, activation="sigmoid")(lstm3)
    bn3 = layers.BatchNormalization()(dense3)

    units4 = 128
    lstm4 = layers.LSTM(units=units4, return_sequences=True)(bn3)
    dense4 = layers.Dense(units=units4, activation="sigmoid")(lstm4)
    bn4 = layers.BatchNormalization()(dense4)

    units5 = 256
    lstm5 = layers.LSTM(units=units5, return_sequences=True)(bn4)
    dense5 = layers.Dense(units=units5, activation="sigmoid")(lstm5)
    bn5 = layers.BatchNormalization()(dense5)

    units6 = 512
    lstm6 = layers.LSTM(units=units6, return_sequences=True)(bn5)
    dense6 = layers.Dense(units=units6, activation="sigmoid")(lstm6)
    bn6 = layers.BatchNormalization()(dense6)

    units6_1 = 512
    lstm6_1 = layers.LSTM(units=units6_1, return_sequences=True)(bn6)
    dense6_1 = layers.Dense(units=units6_1, activation="sigmoid")(lstm6_1)
    bn6_1 = layers.BatchNormalization()(dense6_1)

    units7 = 1024
    lstm7 = layers.LSTM(units=units7, return_sequences=True)(bn6_1)
    dense7 = layers.Dense(units=units7, activation="sigmoid")(lstm7)
    bn7 = layers.BatchNormalization()(dense7)

    units8 = 1024
    lstm8 = layers.LSTM(units=units8, return_sequences=True)(bn7)
    dense8 = layers.Dense(units=units8, activation="sigmoid")(lstm8)
    bn8 = layers.BatchNormalization()(dense8)

    units9 = 2048
    lstm9 = layers.LSTM(units=units9, return_sequences=True)(bn8)
    dense9 = layers.Dense(units=units9, activation="sigmoid")(lstm9)
    bn9 = layers.BatchNormalization()(dense9)

    units9 = input_shape[-1]
    dense10 = layers.Dense(units=units9, activation="sigmoid")(bn9)
    
    output_layer = dense10


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_encoded_4(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model (cases_count, timeframes_count, channels)

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    #input
    input_layer = layers.Input(shape=input_shape)

    units1 = 16
    lstm1_1 = layers.LSTM(units=units1, return_sequences=True)(input_layer)
    bn1_1 = layers.BatchNormalization()(lstm1_1)
    lstm1_2 = layers.LSTM(units=units1, return_sequences=True)(bn1_1)
    bn1_2 = layers.BatchNormalization()(lstm1_2)
    lstm1_3 = layers.LSTM(units=units1, return_sequences=True)(bn1_2)
    bn1_3 = layers.BatchNormalization()(lstm1_3)

    units2 = 32
    lstm2_1 = layers.LSTM(units=units2, return_sequences=True)(bn1_3)
    bn2_1 = layers.BatchNormalization()(lstm2_1)
    lstm2_2 = layers.LSTM(units=units2, return_sequences=True)(bn2_1)
    bn2_2 = layers.BatchNormalization()(lstm2_2)
    lstm2_3 = layers.LSTM(units=units2, return_sequences=True)(bn2_2)
    bn2_3 = layers.BatchNormalization()(lstm2_3)

    units3= 64
    lstm3_1 = layers.LSTM(units=units3, return_sequences=True)(bn2_3)
    bn3_1 = layers.BatchNormalization()(lstm3_1)
    lstm3_2 = layers.LSTM(units=units3, return_sequences=True)(bn3_1)
    bn3_2 = layers.BatchNormalization()(lstm3_2)
    lstm3_3 = layers.LSTM(units=units3, return_sequences=True)(bn3_2)
    bn3_3 = layers.BatchNormalization()(lstm3_3)

    units4 = 128
    lstm4_1 = layers.LSTM(units=units4, return_sequences=True)(bn3_3)
    bn4_1 = layers.BatchNormalization()(lstm4_1)
    lstm4_2 = layers.LSTM(units=units4, return_sequences=True)(bn4_1)
    bn4_2 = layers.BatchNormalization()(lstm4_2)
    lstm4_3 = layers.LSTM(units=units4, return_sequences=True)(bn4_2)
    bn4_3 = layers.BatchNormalization()(lstm4_3)

    units5 = 256
    lstm5_1 = layers.LSTM(units=units5, return_sequences=True)(bn4_3)
    bn5_1 = layers.BatchNormalization()(lstm5_1)
    lstm5_2 = layers.LSTM(units=units5, return_sequences=True)(bn5_1)
    bn5_2 = layers.BatchNormalization()(lstm5_2)
    lstm5_3 = layers.LSTM(units=units5, return_sequences=True)(bn5_2)
    bn5_3 = layers.BatchNormalization()(lstm5_3)

    units6 = 512
    lstm6_1 = layers.LSTM(units=units6, return_sequences=True)(bn5_3)
    bn6_1 = layers.BatchNormalization()(lstm6_1)
    lstm6_2 = layers.LSTM(units=units6, return_sequences=True)(bn6_1)
    bn6_2 = layers.BatchNormalization()(lstm6_2)
    lstm6_3 = layers.LSTM(units=units6, return_sequences=True)(bn6_2)
    bn6_3 = layers.BatchNormalization()(lstm6_3)

    units7 = 1024
    lstm7_1 = layers.LSTM(units=units7, return_sequences=True)(bn6_3)
    bn7_1 = layers.BatchNormalization()(lstm7_1)
    lstm7_2 = layers.LSTM(units=units7, return_sequences=True)(bn7_1)
    bn7_2 = layers.BatchNormalization()(lstm7_2)
    lstm7_3 = layers.LSTM(units=units7, return_sequences=True)(bn7_2)
    bn7_3 = layers.BatchNormalization()(lstm7_3)

    units8 = 2048
    lstm8_1 = layers.LSTM(units=units8, return_sequences=True)(bn7_3)
    bn8_1 = layers.BatchNormalization()(lstm8_1)
    lstm8_2 = layers.LSTM(units=units8, return_sequences=True)(bn8_1)
    bn8_2 = layers.BatchNormalization()(lstm8_2)
    lstm8_3 = layers.LSTM(units=units8, return_sequences=True)(bn8_2)
    bn8_3 = layers.BatchNormalization()(lstm8_3)

    if 0:
        units_out = input_shape[-1]
        dense_out = layers.Dense(units=units_out, activation="sigmoid")(bn8_3)
        output_layer = dense_out
    else:
        units_out = input_shape[-1]
        conv_out = layers.Conv1D(filters=units_out,
            kernel_size=3,
            activation="sigmoid",
            padding="same",
            data_format='channels_last')(bn8_3)
        output_layer = conv_out


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

