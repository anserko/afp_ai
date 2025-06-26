import numpy as np
from PIL import Image


def get_frames(image_data, stride_step, ifPrint = False):
    #split strides
    frames_nmb = image_data.shape[1]//stride_step
    original_shape = image_data.shape
    #slice frames
    #swap axis
    image_data_rsh1 = np.swapaxes(image_data,0,1)  
    #reshape
    image_data_rsh2 = image_data_rsh1.reshape(frames_nmb, stride_step, original_shape[0], original_shape[2])    
    #swap axis back
    image_data_frames = np.swapaxes(image_data_rsh2,1,2)
    
    
    if ifPrint:
        print(f'Total number of frames: {frames_nmb}')
        print(f'Image shape after 1st reshape: {image_data_rsh1.shape}')
        print(f'Image shape after 2st reshape: {image_data_rsh2.shape}')
        print(f'Image shape after final reshape: {image_data_frames.shape}')
    
    return image_data_frames
