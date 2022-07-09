# By Abtin Riasatian, email: abtin.riasatian@gmail.com

# The extract_features function gets a patch directory and a feature directory.
# the function will extract the features of the patches inside the folder
# and saves them in a pickle file of a dictionary mapping patch names to features.


# config variables
patch_dir = "./patches/"
extracted_features_save_adr = "./extracted_features.pickle"
network_weights_address = "./weights/KimiaNetKerasWeights.h5"
network_input_patch_width = 1000
batch_size = 30
img_format = 'jpg'
use_gpu = True

# preprocessing variables
data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]
data_format = "channels_last"


# configuring the device used for inference 
import os

if use_gpu:
    os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# importing libraries
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
from tensorflow.keras.backend import bias_add, constant    

import glob, pickle, skimage.io, pathlib
import numpy as np
from tqdm import tqdm


def preprocessing_fn(input_batch, network_input_patch_width, data_mean, 
                     data_std, data_format):
    
    '''
        Feature extractor preprocessing function, standardizes the input batch 

        Args:
            input_batch: input data batch, a numpy array with 4 dimensions NxWxHxC
            network_input_patch_width: int, size of the input patch, 
                patches are assumed to have the save width and height
            data_mean: list of size 3, mean of each channel of the training data, used for data normalization
            data_std: list of size 3, standard deviation of each channel of the training data, used for data normalization
            data_format: string, specified how each image is represented; for example 'channels_last'


        Returns:
            standardized_input_batch: standardized data batch, same shape as input_batch
    '''

    # getting the input image size
    org_input_size = tf.shape(input_batch)[1]
    
    # standardization
    scaled_input_batch = tf.cast(input_batch, 'float') / 255.
    
    
    # resizing the patches if necessary
    resized_input_batch = tf.cond(tf.equal(org_input_size, network_input_patch_width),
                                lambda: scaled_input_batch, 
                                lambda: tf.image.resize(scaled_input_batch, 
                                                        (network_input_patch_width, network_input_patch_width)))
    
    
    # normalization, this is similar to tf.keras.applications.densenet.preprocess_input()
    mean_tensor = constant(-np.array(data_mean))
    standardized_input_batch = bias_add(resized_input_batch, mean_tensor, data_format)
    standardized_input_batch /= data_std
    
    return standardized_input_batch


# feature extractor initialization function
def kimianet_feature_extractor(network_input_patch_width, weights_address, 
                               data_mean, data_std, data_format):
    
    '''
        This function loads and initializes the KimiaNet feature extractor.
        Args:
            network_input_patch_width: int, size of the input patch, 
                patches are assumed to have the save width and height
            weights_address: string address to network weights file
            data_mean: list of size 3, mean of each channel of the training data, used for data normalization
            data_std: list of size 3, standard deviation of each channel of the training data, used for data normalization
            data_format: string, specified how each image is represented; for example 'channels_last'
        
        Returns:
            kn_feature_extractor_seq: keras model, KimiaNet feature extractor
    '''


    # loading the DenseNet-121 architecture
    dnx = DenseNet121(include_top=False, weights=weights_address, 
                      input_shape=(network_input_patch_width, network_input_patch_width, 3), pooling='avg')

    # connecting the last batch norm layer to a global average pooling layer
    kn_feature_extractor = Model(inputs=dnx.input, outputs=GlobalAveragePooling2D()(dnx.layers[-3].output))
    
    # adding preprocessing to the model sequence
    kn_feature_extractor_seq = Sequential([Lambda(preprocessing_fn, 
                                                  arguments={'network_input_patch_width': network_input_patch_width,
                                                             'data_mean': data_mean, 
                                                             'data_std': data_std, 
                                                             'data_format': data_format}, 
                                                  
                                   input_shape=(None, None, 3), dtype=tf.uint8)])
    
    kn_feature_extractor_seq.add(kn_feature_extractor)
    
    return kn_feature_extractor_seq



# feature extraction function
def extract_features(patch_dir, extracted_features_save_adr, network_weights_address, 
                     network_input_patch_width, batch_size, img_format, data_mean, 
                     data_std, data_format):

    '''
        This function loads the feature extractor, finds the patches available in the specified directory, 
        extracts the features for each patch and saves a dictionary mapping every patch name to their extracted feature.
        
        Args:
            patch_dir: string, directory of patches
            extracted_features_save_adr: string, address to save the extracted features pickle file
            network_weights_address: string address to network weights file
            network_input_patch_width: int, size of the input patch, 
                patches are assumed to have the save width and height
            batch_size: int, network batch_size for inference
            img_format: string, image file type, for example 'jpg'
            data_mean: list of size 3, mean of each channel of the training data, used for data normalization
            data_std: list of size 3, standard deviation of each channel of the training data, used for data normalization
            data_format: string, specified how each image is represented; for example 'channels_last'
    '''
        
    # loading the feature extractor
    feature_extractor = kimianet_feature_extractor(network_input_patch_width, network_weights_address, 
                                                   data_mean, data_std, data_format)
    
    # creating a list of paths for patches available in the specified directory
    patch_adr_list = [pathlib.Path(x) for x in glob.glob(patch_dir+'*.'+img_format)]
    feature_dict = {}

    # extracting features and updating the patch name to feature dictionary
    for batch_st_ind in tqdm(range(0, len(patch_adr_list), batch_size)):
        batch_end_ind = min(batch_st_ind+batch_size, len(patch_adr_list))
        batch_patch_adr_list = patch_adr_list[batch_st_ind:batch_end_ind]
        patch_batch = np.array([skimage.io.imread(x) for x in batch_patch_adr_list])
        batch_features = feature_extractor.predict(patch_batch)
        feature_dict.update(dict(zip([x.stem for x in batch_patch_adr_list], list(batch_features))))
        
        # updating the saved feature dictionary
        with open(extracted_features_save_adr, 'wb') as output_file:
            pickle.dump(feature_dict, output_file, pickle.HIGHEST_PROTOCOL)



# calling the feature extractor function
extract_features(patch_dir, extracted_features_save_adr, network_weights_address, 
                 network_input_patch_width, batch_size, img_format,
                 data_mean, data_std, data_format)

