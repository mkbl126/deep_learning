# coding=utf-8

import os

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Merge, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils.vis_utils import plot_model
from keras.utils import layer_utils
from keras import backend as K

###################################################################

def LeNet5(include_top=True, model_folder=r'.\model') :
    print('>> LeNet5 MODEL. Input shape = (1, 28, 28)')
    model_activation = 'relu'
    num_classes = 10
    init_shape = (1, 28, 28)

    inputs = Input(shape=init_shape)
    x = Conv2D(6, (5, 5), strides=(1, 1), padding='same', activation=model_activation, input_shape=init_shape)(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation=model_activation)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(120, (5, 5), strides=(1, 1), padding='valid', activation=model_activation)(x)
    if include_top :
        x = Flatten()(x)
        x = Dense(84, kernel_initializer='normal', activation=model_activation)(x)
        x = Dropout(0.25)(x)
        x = Dense(num_classes, kernel_initializer='normal')(x)
        x = Activation('softmax')(x)

    model = Model(inputs, x, name='LeNet5')

    if include_top :
        weights_path = os.path.join(model_folder, 'lenet_weights_th_dim_ordering_th_kernels.h5')
    else:
        weights_path = os.path.join(model_folder, 'lenet_weights_th_dim_ordering_th_kernels_notop.h5')
    model.load_weights(weights_path)

    if K.backend() == 'theano':
        layer_utils.convert_all_kernels_in_model(model)

    # if K.image_data_format() == 'channels_first':
    #     if include_top:
    #         maxpool = model.get_layer(name='block5_pool')
    #         shape = maxpool.output_shape[1:]
    #         dense = model.get_layer(name='fc1')
    #         layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')


    # 使用SGD + momentum
    # model.compile里的参数loss就是损失函数(目标函数)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print model.summary()
    # plot_model(model, to_file='model_LeNet5.png')

    return model

###################################################################

def LeNet5_old(include_top=True) :
    print('>> LeNet5 MODEL old. Input shape = (1, 28, 28)')
    model = Sequential()
    model_activation = 'relu'
    num_classes = 10
    init_shape = (1, 28, 28)

    # Layer 1, conv
    model.add(Conv2D(6, (5, 5), strides=(1, 1), padding='same', activation=model_activation, input_shape=init_shape))
    # model.add(Activation(model_activation))
    # Layer 2, pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Layer 3, conv
    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation=model_activation))
    # model.add(Activation(model_activation))
    # Layer 4, pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Layer 5, conv
    model.add(Conv2D(120, (5, 5), strides=(1, 1), padding='valid', activation=model_activation))
    # model.add(Activation(model_activation))

    if include_top :
        # Layer 6, full
        model.add(Flatten())
        model.add(Dense(84, kernel_initializer='normal', activation=model_activation))
        # model.add(Activation(model_activation))
        model.add(Dropout(0.25))

        # Layer 7, full
        model.add(Dense(num_classes, kernel_initializer='normal'))
        model.add(Activation('softmax'))

    if include_top :
        weights_path = 'lenet_weights_th_dim_ordering_th_kernels_old.h5'
    else:
        weights_path = 'lenet_weights_th_dim_ordering_th_kernels_notop.h5'
    model.load_weights(weights_path)

    if K.backend() == 'theano':
        layer_utils.convert_all_kernels_in_model(model)

    #使用SGD + momentum
    #model.compile里的参数loss就是损失函数(目标函数)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print model.summary()
    # plot_model(model, to_file='keras_LeNet5.png')

    return model

###################################################################