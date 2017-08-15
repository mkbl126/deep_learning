
# coding: utf-8

# In[195]:

from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.vis_utils import plot_model
from keras.applications import inception_v3
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
import cv2  # only used for loading the image, you can use anything that returns the image as a np.ndarray

# ## Lets see the cat!

# In[77]:

cat_img = cv2.imread('..\image\cat_blue.png')


# In[78]:

plt.imshow(cat_img)

# In[79]:

# what does the image look like?
print cat_img.shape, cat_img.shape[0], cat_img.shape[1], cat_img.shape[2]


# In[80]:

def adjust_shape_to_show(img) :
    if K.image_data_format() == 'channels_first':
        img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
    print "adjust_shape_to_show ->", K.image_data_format(), img.shape
    return img


# In[81]:

def adjust_shape_to_model(img) :
    if K.image_data_format() == 'channels_first':
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
    print "adjust_shape_to_model ->", K.image_data_format(), img.shape
    return img


# In[82]:

def pre_process(input) :
    img = adjust_shape_to_model(input)
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    input_batch = np.expand_dims(img, axis=0)
    print "input_batch.shape", input_batch.shape
    return input_batch


# In[83]:

# here we get rid of that added dimension and plot the image
def post_process(output_batch) :
    print "output_batch.shape", output_batch.shape
    output = np.squeeze(output_batch, axis=0)
    img = adjust_shape_to_show(output)
    print "img.shape", img.shape
    return img


# ## Load the existed models

# In[149]:

model = vgg16.VGG16(weights='imagenet', include_top=True)
# plot_model(model, to_file='model.png', show_shapes=True)
print model.summary()


# In[189]:

# Get the symbolic outputs of each "key" layer (we gave them unique names).
model_input = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = "block2_conv2"
assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
layer_output = layer_dict[layer_name].output


# In[190]:

input_batch = pre_process(cat_img)


# In[170]:

# output_category = model.predict(input_batch)
# print "output_category.shape", output_category.shape
# print output_category.argmax(), output_category.argmax(axis=-1)
#
#
# # In[174]:
#
# intermediate_layer_model = Model(inputs=model.input, outputs=layer_output)
# output_batch = intermediate_layer_model.predict(input_batch)
# print "output_batch.shape", output_batch.shape


# In[191]:

get_layer_output = K.function([model_input], [layer_output])
output_batch = get_layer_output([input_batch])[0]
print "output_batch.shape", output_batch.shape


# In[192]:

img_features = post_process(output_batch)


# In[201]:

def visualize(img_features) :
    print "img_features.shape", img_features.shape
    subplots = img_features.shape[2]
    columns = 6
    rows = min(16, subplots/columns)
    print rows, columns
    for i in range(min(rows*columns, subplots)) :
        img = img_features[:, :, i]
        plt.subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()
    return


# In[202]:

visualize(img_features)

