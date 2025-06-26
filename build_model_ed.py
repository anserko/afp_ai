import tensorflow as tf
from tensorflow.keras import layers, models


def build_model_1(input_shape):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model

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
    #zeropad = layers.ZeroPadding2D(padding=(0, 1))(input_layer)

    #encoder
    conv1 = layers.Conv2D(filters=8,kernel_size=(3, 3),
                          activation="sigmoid",padding="same",data_format='channels_last')(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(filters=8,kernel_size=(3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pooling1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)

    conv3 = layers.Conv2D(filters=16,kernel_size=(3, 3),activation="sigmoid",
                          padding="same",
                          data_format='channels_last')(pooling1)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(filters=16,kernel_size=(3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    print(f'Last conv layer shape: {conv3.shape}')

    #reshape = layers.Reshape((-1, input_shape[-1]*nx*ny*nz))(conv3)
    flatten = layers.Flatten()(conv3)
    dense_units = 100
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(conv3.shape[1:])(dense2)


    #decoder
    conv4 = layers.Conv2D(filters=16,kernel_size=(3, 3),activation="sigmoid",padding="same",data_format='channels_last')(reshape1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(filters=16,kernel_size=(3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    upsampling1 = layers.UpSampling2D((2,2))(conv4)

    conv6 = layers.Conv2D(filters=8,kernel_size=(3, 3),
                          activation="sigmoid",padding="same",data_format='channels_last')(upsampling1)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(filters=8,kernel_size=(3, 3),activation="sigmoid",padding="same",data_format='channels_last')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    if 0:
        #crop last layer to take into acccount zero padding
        cropping = layers.Cropping2D((0,1))(conv6)

    output = layers.Conv2D(filters=3,kernel_size=(3, 3),activation="sigmoid",
                           padding="same",data_format='channels_last')(conv6)

    output_layer = output


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def build_model_2(input_shape, dense_units, nn_blocks, ifPrint=False):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input section
    input_layer = layers.Input(shape=input_shape)
    #rescale go from [0,255] to [0,1]
    rescale_input = layers.Rescaling(scale=1./255)(input_layer)

    #encoder section
    for i,block in enumerate(nn_blocks):
        filters_nmb, kernel_size, ifBatchNorm, pool_size = block

        if i==0:
            previous_layer = rescale_input
        else:
            previous_layer = block_layer

        #convolutional layer 
        block_layer = layers.Conv2D(
            filters=filters_nmb,      
            kernel_size=kernel_size,
            activation="sigmoid",
            padding="same",
            data_format='channels_last',
                        )(previous_layer)
        
        if ifBatchNorm:
            block_layer = layers.BatchNormalization()(block_layer)
        block_layer = layers.MaxPooling2D(pool_size=pool_size)(block_layer)


    #flatting section
    if ifPrint:
        print(f'Last conv layer shape: {block_layer.shape}')
    flatten = layers.Flatten()(block_layer)
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(block_layer.shape[1:])(dense2)



    for i,block in enumerate(reversed(nn_blocks)):
        filters_nmb, kernel_size, ifBatchNorm, pool_size = block
        if i==0:
            previous_layer = reshape1
        else:
            previous_layer = block_layer

        #convolutional layer 
        block_layer = layers.Conv2D(
            filters=filters_nmb,      
            kernel_size=kernel_size,
            activation="sigmoid",
            padding="same",
            data_format='channels_last',
                        )(previous_layer)
        
        if ifBatchNorm:
            block_layer = layers.BatchNormalization()(block_layer)
        block_layer = layers.UpSampling2D(size=pool_size)(block_layer)


    #output section
    output = layers.Conv2D(filters=3,kernel_size=(3, 3),activation="sigmoid",
                           padding="same",data_format='channels_last')(block_layer)
    #rescale go from [0,1] to [0,255]
    rescale_output = layers.Rescaling(scale=255)(output)
    output_layer = rescale_output


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def build_model_3(input_shape, dense_units, nn_blocks, ifPrint=False):
    """
    Create neural network model

    Parameters
    ----------
    input_shape : tuple
        input shape for the model

    Returns
    -------
    model : tensorflow.keras.model
        neural network model
    """ 

    '''
    zero-padding and cropping
    https://github.com/keras-team/keras/issues/4818
    '''

    #input section
    input_layer = layers.Input(shape=input_shape)
    #rescale go from [0,255] to [0,1]
    rescale_input = layers.Rescaling(scale=1./255)(input_layer)

    #encoder section
    for i,block in enumerate(nn_blocks):
        filters_nmb_list, kernel_size, ifBatchNorm, pool_size = block
        #check to incorporate build_model_2, when a number is submitted instead of list for filters_nmb_list
        if type(filters_nmb_list)!=list:
            filters_nmb_list = [filters_nmb_list]

        if i==0:
            previous_layer = rescale_input
        else:
            previous_layer = block_layer

        for j, filters_nmb in enumerate(filters_nmb_list):
            if j!=0:
                previous_layer = block_layer
            
            #convolutional layer 
            block_layer = layers.Conv2D(
                filters=filters_nmb,      
                kernel_size=kernel_size,
                activation="sigmoid",
                padding="same",
                data_format='channels_last',
                            )(previous_layer)
        
        if ifBatchNorm:
            block_layer = layers.BatchNormalization()(block_layer)
        block_layer = layers.MaxPooling2D(pool_size=pool_size)(block_layer)


    #flatting section
    if ifPrint:
        print(f'Last conv layer shape: {block_layer.shape}')
    flatten = layers.Flatten()(block_layer)
    dense1 = layers.Dense(units=dense_units, activation="sigmoid")(flatten)
    dense2 = layers.Dense(units=flatten.shape[-1], activation="sigmoid")(dense1)
    reshape1 = layers.Reshape(block_layer.shape[1:])(dense2)



    for i,block in enumerate(reversed(nn_blocks)):
        filters_nmb_list, kernel_size, ifBatchNorm, pool_size = block
        #check to incorporate build_model_2, when a number is submitted instead of list for filters_nmb_list
        if type(filters_nmb_list)!=list:
            filters_nmb_list = [filters_nmb_list]
        if i==0:
            previous_layer = reshape1
        else:
            previous_layer = block_layer

        for j, filters_nmb in enumerate(filters_nmb_list):
            if j!=0:
                previous_layer = block_layer
            #convolutional layer 
            block_layer = layers.Conv2D(
                filters=filters_nmb,      
                kernel_size=kernel_size,
                activation="sigmoid",
                padding="same",
                data_format='channels_last',
                            )(previous_layer)
        
        if ifBatchNorm:
            block_layer = layers.BatchNormalization()(block_layer)
        block_layer = layers.UpSampling2D(size=pool_size)(block_layer)


    #output section
    output = layers.Conv2D(filters=3,kernel_size=(3, 3),activation="sigmoid",
                           padding="same",data_format='channels_last')(block_layer)
    #rescale go from [0,1] to [0,255]
    rescale_output = layers.Rescaling(scale=255)(output)
    output_layer = rescale_output


    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model