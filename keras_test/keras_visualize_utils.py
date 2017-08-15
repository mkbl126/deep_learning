
# coding: utf-8

import math
import numpy as np
import matplotlib.pyplot as plt 

from keras import backend as K

def normalize(x) :
    y = np.copy(x)
    y -= x.min()
    y /= (x.max()-x.min())
    return y 

def de_normalize(x, ori_x) :
    y = np.copy(x)
    y *= (ori_x.max()-ori_x.min())
    y += ori_x.min()
    return y 

def adjust_shape_to_show(img) :
    if K.image_data_format() == 'channels_first':
        img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
    print "adjust_shape_to_show ->", K.image_data_format(), img.shape
    return img

def adjust_shape_to_model(img) :
    if K.image_data_format() == 'channels_first':
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
    print "adjust_shape_to_model ->", K.image_data_format(), img.shape
    return img
    
def pre_process_one_batch(input) :
    norm_input = normalize(input)
    img = adjust_shape_to_model(norm_input)
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    input_batch = np.expand_dims(img, axis=0)
    print "input_batch.shape", input_batch.shape
    return input_batch

def post_process_one_batch(output_batch, ori_input) :
    print "output_batch.shape", output_batch.shape
    output = np.squeeze(output_batch, axis=0)
    normed_img = adjust_shape_to_show(output)
    img = de_normalize(normed_img, ori_input)
    print "img.shape", img.shape
    return img
    
def visualize_in_flat(img_data) :
    print "Visualize image in flat mdoe. Flat :", img_data.shape[-1]
    print "img_data.shape", img_data.shape, len(img_data.shape)
    subplots = img_data.shape[-1]
    columns = 6
    rows = min(16, int(math.ceil(float(subplots)/float(columns))))
#     print subplots, rows, columns
    for i in range(min(rows*columns, subplots)) :
        img = img_data[:, :, i].astype('uint8')
        plt.subplot(rows, columns, i+1)
#         print img
        plt.imshow(img)
    plt.show()
    return

def visualize_in_overlay(img_data) :
    print "Visualize image in overlay mdoe. Overlay :", img_data.shape[-1]
    print "img_data.shape", img_data.shape, len(img_data.shape)
    img = np.copy(img_data[:, :, 0])
    for i in range(1, img_data.shape[-1]) :
        img += img_data[:, :, i]
    img /= img_data.shape[-1]
    plt.imshow(img)
    plt.show()
    return

def visualize_img(img_data, bFlat=False) :
    print "Visualize image. Feature maps %d, Size %d*%d" \
      % (img_data.shape[-1], img_data.shape[0], img_data.shape[1])
    if img_data.shape[-1]<4 or bFlat==True :
        visualize_in_flat(img_data)
    else :
        visualize_in_overlay(img_data)
    return
    
def visualize_layer_weights(layer_weights, bFlat=False) :
    weights = layer_weights[0]
    bias = layer_weights[1]
    print "Visualize weights. Layers Features %d, Dim %d, Size %d*%d" \
          % (weights.shape[-1], weights.shape[-2], weights.shape[0], weights.shape[1])
    for i_feature in range(min(16, weights.shape[-1])) :
        weights_img = (normalize(weights[:, :, :, i_feature])*255).astype('uint8')
        print "Layer Weights:", i_feature, weights_img.shape
#         print weights_img
        visualize_img(weights_img, bFlat)
    return