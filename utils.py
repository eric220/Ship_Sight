import csv
from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D
from keras.layers import Conv2DTranspose, Activation, UpSampling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import optimizers
from keras import applications
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

def get_targets(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        targets_names = [1]
        targets_without_repeats = []
        for row in reader:
            if row[0] != targets_names[-1]:
                targets_names.append(row[0])
                if row[1] == "":
                    targets_without_repeats.append(0)
                else:
                    targets_without_repeats.append(1)
    targets_names = targets_names[1:]
    assert len(targets_names) == len(targets_without_repeats) 
    return (targets_names, targets_without_repeats) 

def path_to_tensor(img_path):
    # loads rgb image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
   # for path in img_paths[:5]:
    list_of_tensors = [path_to_tensor('train_files/%s' % img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)  

def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

def get_model():
    img_width, img_height, channels  = 224, 224, 3

    model = applications.VGG19(weights = 'imagenet', include_top = False, input_shape = (img_width, img_height, channels))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(1, activation = 'sigmoid')(x)
    final_model = Model(input = model.input, output = predictions)
    
    final_model.compile(loss='binary_crossentropy', optimizer = optimizers.SGD(lr=0.0005, momentum=0.9), metrics = ['accuracy'])
    #final_model.load_weights('gap4_weights_improvement.hdf5')
    
    return final_model

def get_activation_model():
    img_width, img_height = 224,224
    channels = 3
    num_classes = 2
    
    GAP_AM_VGG_model = applications.VGG19(weights= 'imagenet', 
                                          include_top = False, input_shape = (img_width, img_height, channels))

    x = GAP_AM_VGG_model.output
    act_map = Activation('relu', name='activation_map')(x)
    x = GlobalAveragePooling2D()(act_map)
    predictions = Dense(num_classes, activation = 'softmax', name='predictions')(x)

    gap_AM_model_final = Model(input = GAP_AM_VGG_model.input, outputs = [predictions, act_map])
    
    return gap_AM_model_final

def get_upsample_model():
    img_width, img_height = 224,224
    channels = 3
    num_classes = 2
    
    GAP_AM_VGG_model = applications.VGG19(weights= 'imagenet', 
                                          include_top = False, input_shape = (img_width, img_height, channels))

    x = GAP_AM_VGG_model.output
    act_map = Activation('relu', name='activation_map')(x)
    up_sample = UpSampling2D(size=(2, 2), data_format=None)(act_map)
    x = GlobalAveragePooling2D()(act_map)
    predictions = Dense(num_classes, activation = 'softmax', name='predictions')(x)

    gap_AM_model_final = Model(input = GAP_AM_VGG_model.input, outputs = [predictions, up_sample])
    
    return gap_AM_model_final

#upsampler = Sequential([UpSampling2DBilinear((256, 256))])
#upsampled = upsampler.predict(images)