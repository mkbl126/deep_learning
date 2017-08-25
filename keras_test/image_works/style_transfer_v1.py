# coding: utf-8
# # Convolutional neural networks for artistic style transfer

from __future__ import print_function

import os.path
import time
from PIL import Image
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.misc import imsave

###############################################################################

def reshape_img(x) :
    if K.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[2], x.shape[0], x.shape[1])
    else:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    return x

def load_image_to_expanded_norm_array(image_path) :
    image_data = Image.open(image_path)
    image_data = image_data.resize((height, width))
    image_array = np.asarray(image_data, dtype='float32')
    image_array = reshape_img(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    image_array[:, :, :, 0] -= 103.939
    image_array[:, :, :, 1] -= 116.779
    image_array[:, :, :, 2] -= 123.68
    image_array = image_array[:, :, :, ::-1]
    return image_array

def MSE(y, y_hat) :
    return K.mean(K.square(y-y_hat))

def get_content_loss(model, layer_names, img_array) :
    content_losses = K.variable(0.)
    for layer_name in layer_names :
        layer_output = model.get_layer(layer_name).output
        get_layer_output = K.function([model.input], [layer_output])
        img_content = get_layer_output([img_array])[0]
        img_combined = layer_output
        content_losses += MSE(img_content, img_combined)
    content_loss = content_losses / len(layer_names)
#     print(type(content_loss), len(layer_names))
    return content_loss


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def calc_style_loss(y, y_hat) :
    channels = 3
    size = height * width
    return K.sum(K.square(y - y_hat)) / (4. * (channels ** 2) * (size ** 2))

def get_style_loss(model, layer_names, img_array) :
    style_losses = K.variable(0.)
    for layer_name in layer_names :
        layer_output = model.get_layer(layer_name).output
        get_layer_output = K.function([model.input], [layer_output])
        img_style = get_layer_output([img_array])[0]
        img_combined = layer_output
        gram_img_style = gram_matrix(img_style[0, :, :, :])
        gram_img_combined = gram_matrix(img_combined[0, :, :, :])
        style_losses += (calc_style_loss(gram_img_style, gram_img_combined))
    style_loss = style_losses / len(layer_names)
#     print(type(style_loss), len(layer_names))
    return style_loss

def get_denoise_loss(x):
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

###############################################################################

height = 512
width = 512

# combination_image_shape = (height, width, 3)
# print(type(combination_image_shape))
# combination_image = K.placeholder((1, height, width, 3))
# model = VGG16(input_tensor=combination_image, weights='imagenet', include_top=False)
# layers = dict([(layer.name, layer.output) for layer in model.layers])
#
# print("="*96)
# for name, output in layers.items() :
#     print(name, output)
# print("="*96)

content_layer_names = ['block2_conv2']
style_layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

weight_content = 1.5
weight_style = 10.0
weight_denoise = 0.3
src1_image_data = load_image_to_expanded_norm_array(r"C:\DL\images\style_transfer\contents\tower.jpg")
src2_image_data = load_image_to_expanded_norm_array(r"C:\DL\images\style_transfer\styles\block.jpg")
src1_image_data = np.random.random((1, height, width, 3))
src2_image_data = np.random.random((1, height, width, 3))

tf_session = K.get_session()
src1_image = K.variable(value=src1_image_data, dtype='float32')
src2_image = K.variable(value=src2_image_data, dtype='float32')
# src1_image = np.random.random((1, height, width, 3))
# src2_image = np.random.random((1, height, width, 3))
print(K.dtype(src1_image))
# src1_image.eval(session=tf_session)

# src1_image = K.placeholder((1, height, width, 3), dtype='float32')
# src2_image = K.placeholder((1, height, width, 3), dtype='float32')

combination_image = K.placeholder((1, height, width, 3))
model = VGG16(input_tensor=combination_image, weights='imagenet', include_top=False)
loss_content = get_content_loss(model, content_layer_names, src1_image.eval(session=tf_session))
loss_style = get_style_loss(model, style_layer_names, src2_image.eval(session=tf_session))
loss_denoise = get_denoise_loss(combination_image)
loss = weight_content*loss_content + weight_style*loss_style + weight_denoise*loss_denoise
grad = K.gradients(loss, combination_image)[0]
optimize_image = K.function([src1_image, src2_image, combination_image], [loss_content, loss_style, loss_denoise, loss, grad])


step_size = 10
num_iterations = 10

folder_src1 = r"C:\DL\images\style_transfer\contents"
folder_src2 = r"C:\DL\images\style_transfer\styles"
folder_dest = r"C:\DL\images\style_transfer\combinations"

files_src1 = os.listdir(folder_src1)
for file_src1 in files_src1 :
    src1_image_path = os.path.join(folder_src1, file_src1)
    if os.path.isfile(src1_image_path) :
        src1_array = load_image_to_expanded_norm_array(src1_image_path)

        files_src2 = os.listdir(folder_src2)
        for file_src2 in files_src2 :
            src2_image_path = os.path.join(folder_src2, file_src2)
            if os.path.isfile(src2_image_path) :
                src2_array = load_image_to_expanded_norm_array(src2_image_path)
                print(src2_array)
                # # x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

                # src1_image_data = src2_array
                # src2_image_data = src2_array
                K.update(src1_image, src1_image_data)
                K.update(src2_image, src2_image_data)
                x = src1_array
                print(src2_image.eval(session=tf_session))

                combination_image_name = "style_" + file_src1.split(".")[0] + "_" + file_src2.split(".")[0] + ".jpg"
                combination_image_path = os.path.join(folder_dest, combination_image_name)
                print("Style Transfer -> [%s] + [%s] = [%s]" % (file_src1, file_src2, combination_image_name))
                # print(height, width, src1_array.shape, src2_array.shape, combination_image.shape, x.shape)

                for i in range(num_iterations) :
                    start_time = time.time()
                    loss_content, loss_style, loss_denoise, loss, grad = optimize_image([src1_image.eval(session=tf_session), src2_image.eval(session=tf_session), x])
                    step_size_scaled = step_size / (np.std(grad) + 1e-8)
                    x -= grad*step_size_scaled
                    x = np.clip(x, 0.0, 255.0)
                    end_time = time.time()
                    # print("Iteration %02d in %ds. L_content %f, L_style %f, L_denoise %f, Loss %f" %
                    #       (i, end_time-start_time, loss_content, loss_style, loss_denoise, loss))

                x = x.reshape((height, width, 3))
                x = x[:, :, ::-1]
                x[:, :, 0] += 103.939
                x[:, :, 1] += 116.779
                x[:, :, 2] += 123.68
                x = np.clip(x, 0, 255).astype('uint8')

                imsave(combination_image_path, x)
                # print("Saved at: ", combination_image_path)

                # K.clear_session()