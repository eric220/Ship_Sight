
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils
from PIL import Image
from keras.models import Model
from keras.applications import VGG19
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras import optimizers
from keras.utils import to_categorical
from scipy.ndimage import zoom

###get file names and targets
def get_targets():
    filename = "train_ship_segmentation/train_ship_segmentations.csv"
    train_files, train_targets = utils.get_targets(filename)
    return train_files, train_targets

###get tensor from file
def get_tensors():
    eval_tensors = utils.paths_to_tensor(train_files[-100:]).astype('float32')/255
    return eval_tensors

###get tensor for dispaly
def get_tensor(num):
    tensor_to_view = eval_tensors[num].reshape(1, 224, 224, 3)
    return tensor_to_view

###get prediction and activation map from model
def get_predictions(tensor):
    predition, activation_map = act_model.predict(tensor_to_view)
    return predition, activation_map

###generate heat map
def get_mask(act_map, act=None):
    mat_for_mult = zoom(act_map, (1,32, 32, 1), order=2, mode='reflect', prefilter =False)
    weights_for_mult = np.array(act_model.layers[-1].get_weights()[0])
    
    final_output = np.dot(mat_for_mult, weights_for_mult)
    final_output2 = final_output[0, :, :, 1]
    
    return final_output2

###reshape tensor for display
def get_tensor_to_view(tensor_to_view):
    tensor_img = np.array(tensor_to_view)
    tensor_img = tensor_img[0, : , : , 0]
    
    return tensor_img

###display image with heat map
def compare_images(tens):
    failed_tensor = eval_tensors[tens]
    data = np.reshape(failed_tensor, (224,224,3))
    plt.imshow(data, interpolation='nearest', aspect = 'auto')  
    plt.imshow(final_output, cmap='jet', alpha=.5, aspect = 'auto')
    plt.show()

    
if __name__ == '__main__':   
    
    ###Setup
    train_files, train_targets = get_targets()
    eval_tensors = get_tensors()
    act_model = utils.get_activation_model()
    act_model.compile(loss='categorical_crossentropy', optimizer = optimizers.SGD(lr=0.0005, momentum=0.9), metrics = ['accuracy'])
    act_model.load_weights('gap_AM_weights_improvement.hdf5')
    
    ###Get tensor to display, predict and heatmap
    tensor_choice = ''
    while tensor_choice != 'q':
        tensor_choice = raw_input('Pick a number between 1 and 100. Input q to quit')
        
        tensor_to_get = int(tensor_choice)
        if tensor_to_get <= 99 and tensor_to_get >= 0:
            #tensor_to_get = int(tensor_to_get)
            tensor_to_view = get_tensor(tensor_to_get)
            predition, activation_map = get_predictions(tensor_to_view)
            print("Prediction:", predition[0][1])
            if predition[0][1] >= 0.5:
                final_output = get_mask(activation_map)
                tensor_img = get_tensor_to_view(tensor_to_view)
                compare_images(tensor_to_get)   
        else: 
            tensor_to_get = ''
    





