# coding=utf-8

import os

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Merge, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils.vis_utils import plot_model
from keras.utils import layer_utils
from keras import backend as K

###################################################################

def AlexNet(include_top=True, model_folder=r'.\model') :
    print('>> AlexNet Model. Input shape = (3, 32, 32)')
    model_pool_size = (2, 2)
    model_pool_strides = (2, 2)
    model = Sequential()

    init_shape = (3, 32, 32)
    model_activation = "relu"
    num_classes = 10

    # Layer 1, conv+pooling
    model.add(Conv2D(96, (11, 11), strides=(1, 1), padding='same', input_shape=init_shape))
    model.add(Activation(model_activation))
    model.add(MaxPooling2D(pool_size=model_pool_size, strides=model_pool_strides))
    model.add(BatchNormalization(axis=1))

    # Layer 2, conv+pooling
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same'))
    model.add(Activation(model_activation))
    model.add(MaxPooling2D(pool_size=model_pool_size, strides=model_pool_strides))
    model.add(BatchNormalization(axis=1))

    # Layer 3, conv+pooling
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation(model_activation))

    # Layer 4, conv+pooling
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation(model_activation))

    # Layer 5, conv+pooling
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation(model_activation))
    model.add(MaxPooling2D(pool_size=model_pool_size, strides=model_pool_strides))

    if include_top :
        # Layer 6, full
        model.add(Flatten())
        model.add(Dense(4096, kernel_initializer='normal'))
        model.add(Activation(model_activation))
        model.add(Dropout(0.5))

        # Layer 7, full
        model.add(Dense(4096, kernel_initializer='normal'))
        model.add(Activation(model_activation))
        model.add(Dropout(0.5))

        # Layer 8, full
        model.add(Dense(num_classes, kernel_initializer='normal'))
        model.add(Activation('softmax'))

    if include_top :
        weights_path = os.path.join(model_folder, 'alexnet_weights_th_dim_ordering_th_kernels.h5')
    else:
        weights_path = os.path.join(model_folder, 'alexnet_weights_th_dim_ordering_th_kernels_notop.h5')
    model.load_weights(weights_path)

    if K.backend() == 'theano':
        layer_utils.convert_all_kernels_in_model(model)

    # 使用SGD + momentum
    # model.compile里的参数loss就是损失函数(目标函数)
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print(model.summary())
    # plot_model(model, to_file='AlexNet.png')

    # model.save_weights('alexnet_weights_th_dim_ordering_th_kernels.h5')
    return model

###################################################################