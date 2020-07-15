import glob
from tqdm import tqdm
from PIL import ImageFile  
from keras.preprocessing import image 
import numpy as np
import utils
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import sys

#returns 4D tensor 
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

#returns tensors for imagfilepaths
def paths_to_tensor(img_paths):
    if len(img_paths) > 1:
        list_of_tensors = [path_to_tensor('%s' % img_path) for img_path in tqdm(img_paths)]
    else:
        list_of_tensors = [path_to_tensor(img_paths[0]) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)  

def get_mask(act_map, act=None):
    weights_for_mult = np.array(model.layers[-1].get_weights()[0])
    final_output = np.dot(act_map, weights_for_mult)
    final_output2 = final_output[0, :, :, 0]
    out = zoom(final_output2, (4.15, 4.15), order=2, mode='reflect', prefilter =False)
    return out

def get_tensor_to_view(tensor_to_view):
    tensor_img = np.array(tensor_to_view)
    tensor_img = tensor_img[0, : , : , 0]
    return tensor_img

def compare_images(tens, final_output, pred):
    tens = np.reshape(tens, (224,224,3))
    f, (ax1, ax2) = plt.subplots(1,2, figsize = (10,10))
    ax1.imshow(final_output, cmap='jet')
    ax2.imshow(tens, interpolation='nearest')
    plt.title('Prediction:{}' .format(pred[0][0]))
    plt.xticks([])
    plt.yticks([])
    plt.show(block = False) 
    raw_input('Press Enter to close')
    plt.close('all')
    return 
    
def get_model():
    model = utils.get_activation_model()
    model.compile(loss='categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
    model.load_weights('ship_weights.hdf5')
    return model

model = get_model()

if __name__ == "__main__":
    file_names = glob.glob('test_img/*.jpg')
    eval_tensors = paths_to_tensor([file_names[0]]).astype('float32')/255
    prediction, activation_map = model.predict(eval_tensors)
    print("Prediction:{}" .format(prediction))
    final_output = get_mask(activation_map)
    tensor_img = get_tensor_to_view(eval_tensors)
    compare_images(eval_tensors, final_output, prediction)