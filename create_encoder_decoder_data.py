import numpy as np
from PIL import Image
from get_frames import get_frames


def create_encoder_decoder_data(file_list, path_list, stride_step, angle_settings_list, ifFlip=True, dtype = 'float64', numb_decimals=False, norm_factor=255):


    angle_min, angle_max, angle_step = angle_settings_list
    angle_list = np.arange(angle_min, 1.1*angle_max, angle_step)

    for file in file_list:
        #get path
        for path in path_list:
            index = 4
            if file[:index] == path[:index]:
                break
        print(file, path)

        #print(f'Opened file: {file}')
        img = Image.open(f'{path}//{file}')
        if ifFlip:
            #transpose the original image
            img_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
        #convert to array
        image_data_original = np.array(img, dtype=dtype)

        for i, angle in enumerate(angle_list):
            #rotate the image
            img_rotated = img.rotate(angle)
            angle_flip = angle*(-1)
            if ifFlip:
                img_flip_rotated = img_flip.rotate(angle_flip)
            #convert to array
            image_rotated_data = np.array(img_rotated, dtype=dtype)
            if ifFlip:
                image_flip_rotated_data = np.array(img_flip_rotated, dtype=dtype)
            #cutoff the end
            image_rotated_data = image_rotated_data[:,:-96,:]
            if ifFlip:
                image_flip_rotated_data = image_flip_rotated_data[:,:-96,:]

            if 1:
                #fill black regions with background colour  
                #color used for replacement from top left corner
                r2,g2,b2 = image_data_original[0,0]    
                #create mask where pixels are black
                r1, g1, b1 = image_rotated_data[:,:,0], image_rotated_data[:,:,1], image_rotated_data[:,:,2]
                mask = (r1 == 0) & (g1 == 0) & (b1 == 0)
                #replace black pixels
                image_rotated_data[:,:,:3][mask] = [r2, g2, b2]
                if ifFlip:
                    #repeat for flipped image
                    #create mask where pixels are black
                    r1, g1, b1 = image_flip_rotated_data[:,:,0], image_flip_rotated_data[:,:,1], image_flip_rotated_data[:,:,2]
                    mask = (r1 == 0) & (g1 == 0) & (b1 == 0)
                    #replace black pixels
                    image_flip_rotated_data[:,:,:3][mask] = [r2, g2, b2]

            image_rotated_data_frames = get_frames(image_rotated_data, stride_step, ifPrint = False)
            #normalise
            if norm_factor!=1:
                image_rotated_data_frames = image_rotated_data_frames/norm_factor
            if numb_decimals:
                image_rotated_data_frames = np.around(image_rotated_data_frames, numb_decimals)
            if ifFlip:
                image_flip_rotated_data_frames = get_frames(image_flip_rotated_data, stride_step, ifPrint = False)
                #normalise
                if norm_factor!=1:
                    image_flip_rotated_data_frames = image_flip_rotated_data_frames/norm_factor
                if numb_decimals:
                    image_flip_rotated_data_frames = np.around(image_flip_rotated_data_frames, numb_decimals)
            
            #join results
            if 'frames_data' in locals():
                if ifFlip:
                    frames_data = np.concatenate((frames_data,
                                                  image_rotated_data_frames, 
                                                  image_flip_rotated_data_frames), 
                                                 axis=0)
                else:
                    frames_data = np.concatenate((frames_data,
                                                  image_rotated_data_frames), 
                                                 axis=0)
            else:
                if ifFlip:
                    frames_data = np.concatenate((image_rotated_data_frames, 
                                                  image_flip_rotated_data_frames), 
                                                 axis=0)
                else:
                    frames_data = image_rotated_data_frames
                    
    return frames_data




def rotate_image(image, angle, dtype, crop_horizontal, fill_colour):
    #rotate the image
    img_rotated = image.rotate(angle) 
    #convert to array
    image_rotated_data = np.array(img_rotated, dtype=dtype)

    #cutoff/crop the end from the right
    image_rotated_data = image_rotated_data[:,:-crop_horizontal,:]

    #fill black regions with background colour    
    #colour to be used to replace black regions occuring due to image rotation
    r2, g2, b2 = fill_colour
    #create mask where pixels are black
    r1, g1, b1 = image_rotated_data[:,:,0], image_rotated_data[:,:,1], image_rotated_data[:,:,2]
    mask = (r1 == 0) & (g1 == 0) & (b1 == 0)
    #replace black pixels
    image_rotated_data[:,:,:3][mask] = [r2, g2, b2]

    return image_rotated_data


def create_augmented_frames(image, stride_step, angle_list, crop_horizontal, dtype = 'float64', norm_factor=255, ifPrint=False):

    
    #getting the colour of the top left corner of the original image to fill black regions occuring due to image rotation
    #convert to array
    image_data_original = np.array(image, dtype=dtype)
    #color used for replacement from top left corner
    r2,g2,b2 = image_data_original[0,0]
    #delete image_data_original array as it is not required anymore (memory save) 
    del image_data_original

    for i, angle in enumerate(angle_list):
        #rotate the image
        image_rotated_data = rotate_image(image, angle, dtype, crop_horizontal, fill_colour = (r2,g2,b2))
        #get frames
        image_rotated_data_frames = get_frames(image_rotated_data, stride_step, ifPrint = False)
        #normalise
        if norm_factor!=1:
            image_rotated_data_frames = image_rotated_data_frames/norm_factor
        
        #join results
        if i!=0:
            frames_data = np.concatenate((frames_data,
                                          image_rotated_data_frames), 
                                         axis=0)
        else:
            frames_data = image_rotated_data_frames
                    
    return frames_data
    

def process_image(file_path, stride_step, angle_list, crop_horizontal, ifFlip, dtype = 'float64', norm_factor=255, ifPrint=False):

    if ifPrint:
        print(file_path)
    image = Image.open(file_path)
    frames_data = create_augmented_frames(
        image,
        stride_step,
        angle_list,
        crop_horizontal,
        dtype=dtype,
        norm_factor=norm_factor,
        ifPrint=ifPrint
        )

    if ifFlip:
        #transpose the original image
        image_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
        angle_flip_list = angle_list*(-1)
        frames_data_flipped = create_augmented_frames(
            image_flip,
            stride_step,
            angle_flip_list,
            crop_horizontal,
            dtype = dtype,
            norm_factor=norm_factor,
            ifPrint=ifPrint
            )
        frames_data = np.concatenate((frames_data,frames_data_flipped),axis=0)

    return frames_data


def process_dataset(file_path_list, stride_step, angle_list, crop_horizontal, ifFlip, dtype = 'float64', norm_factor=255, ifPrint=False):
    for i, file_path in enumerate(file_path_list):
        
        if ifPrint:
            print(file_path)
        
        frames_single_data = process_image(
            file_path, 
            stride_step, 
            angle_list,
            crop_horizontal,
            ifFlip, dtype = dtype,
            norm_factor=norm_factor,
            ifPrint=ifPrint 
        )
        
        if i:
            frames_data = np.concatenate((frames_data,frames_single_data),axis=0)
        else:
            frames_data = frames_single_data

    return frames_data

def get_full_path(image_list, folder_list):

    file_path_list = [f'{folder}//{file}' for file in image_list for folder in folder_list if file[:4]==folder[:4]]
    return file_path_list


