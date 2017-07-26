# coding=utf-8

"""
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
"""

# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

# import os
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"

import time
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Merge, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model

import keras_data

###################################################################

def ConvBlock(input, kernels_size, kernels_strides, model_activation) :
    kernels0_paras = list(kernels_size)
    kernel0_1_size = kernels0_paras[0]
    kernel0_2_size = kernels0_paras[1]
    kernel1_size = kernel0_3_size = kernels0_paras[2]
    tower_0 = Conv2D(kernel0_1_size, (1, 1), strides=kernels_strides, padding='valid')(input)
    tower_0 = BatchNormalization(axis=1)(tower_0)
    tower_0 = Activation(model_activation)(tower_0)
    tower_0 = Conv2D(kernel0_2_size, (3, 3), padding='same')(tower_0)
    tower_0 = BatchNormalization(axis=1)(tower_0)
    tower_0 = Activation(model_activation)(tower_0)
    tower_0 = Conv2D(kernel0_3_size, (1, 1), padding='same')(tower_0)
    tower_0 = BatchNormalization(axis=1)(tower_0)
    tower_1 = Conv2D(kernel1_size, (1, 1), strides=kernels_strides, padding='valid')(input)
    tower_1 = BatchNormalization(axis=1)(tower_1)
    output = keras.layers.add([tower_0, tower_1])
    output = Activation(model_activation)(output)
    return output

def IdentityBlock(input, kernels0, model_activation) :
    kernels0_paras = list(kernels0)
    kernel0_1_size = kernels0_paras[0]
    kernel0_2_size = kernels0_paras[1]
    kernel0_3_size = kernels0_paras[2]
    tower_0 = Conv2D(kernel0_1_size, (1, 1), padding='same')(input)
    tower_0 = BatchNormalization(axis=1)(tower_0)
    tower_0 = Activation(model_activation)(tower_0)
    tower_0 = Conv2D(kernel0_2_size, (3, 3), padding='same')(tower_0)
    tower_0 = BatchNormalization(axis=1)(tower_0)
    tower_0 = Activation(model_activation)(tower_0)
    tower_0 = Conv2D(kernel0_3_size, (1, 1), padding='same')(tower_0)
    tower_0 = BatchNormalization(axis=1)(tower_0)
    output = keras.layers.add([tower_0, input])
    output = Activation(model_activation)(output)
    return output

def ResNet_cifar10(X_train, Y_train, X_test, Y_test, input_shape, num_classes, model_batch_size, model_epochs, model_activation) :
    print '>> ResNet_cifar10 TRAINT AND TEST'
    size_1 = (1, 1)
    strides_1 = (1, 1)
    size_3 = (3, 3)
    strides_2 = (2, 2)

    g0_input = Input(shape=init_shape)

    # Group 1, conv+bn+act+pooling, output 32*32
    g1_padding1 = ZeroPadding2D((1, 1))(g0_input)
    g1_conv1    = Conv2D(16, (3, 3), strides=(1, 1), padding='valid')(g1_padding1)
    g1_bn1      = BatchNormalization(axis=1)(g1_conv1)
    g1_output   = g1_act1 = Activation(model_activation)(g1_bn1)
    # g1_padding2 = ZeroPadding2D((1, 1))(g1_act1)
    # g1_pooling1 = MaxPooling2D(pool_size=size_3, strides=strides_2)(g1_padding2)

    # Group 2, 1*ConvBlock + 2*IdentityBlock, output 32*32
    kernels_size = (16, 16, 64)
    kernels_strides = (1, 1)
    g2_conv_block1 = ConvBlock(g1_output, kernels_size, kernels_strides, model_activation)
    g2_identity_block1 = IdentityBlock(g2_conv_block1, kernels_size, model_activation)
    g2_identity_block2 = IdentityBlock(g2_identity_block1, kernels_size, model_activation)
    g2_output = g2_identity_block3 = IdentityBlock(g2_identity_block2, kernels_size, model_activation)

    # Group 3, 1*ConvBlock + 3*IdentityBlock, output 16*16
    kernels_size = (32, 32, 128)
    kernels_strides = (2, 2)
    g3_conv_block1 = ConvBlock(g2_output, kernels_size, kernels_strides, model_activation)
    g3_identity_block1 = IdentityBlock(g3_conv_block1, kernels_size, model_activation)
    g3_identity_block2 = IdentityBlock(g3_identity_block1, kernels_size, model_activation)
    g3_output = g3_identity_block3 = IdentityBlock(g3_identity_block2, kernels_size, model_activation)

    # Group 4, 1*ConvBlock + 5*IdentityBlock, output 8*8
    kernels_size = (64, 64, 256)
    kernels_strides = (2, 2)
    g4_conv_block1 = ConvBlock(g3_output, kernels_size, kernels_strides, model_activation)
    g4_identity_block1 = IdentityBlock(g4_conv_block1, kernels_size, model_activation)
    g4_identity_block2 = IdentityBlock(g4_identity_block1, kernels_size, model_activation)
    g4_output = g4_identity_block3 = IdentityBlock(g4_identity_block2, kernels_size, model_activation)

    # Group 5, avg pool, output 1*1
    g5_pooling1   = AveragePooling2D((8, 8), strides=strides_1, padding='valid')(g4_output)

    # Group 6, avg pool, output 7*7
    g6_flatten = Flatten()(g5_pooling1)
    g6_output  = Dense(num_classes, kernel_initializer='normal', activation='softmax')(g6_flatten)

    model = Model(inputs=g0_input, outputs=[g6_output], name="ResNet50_cifar10")

    # use SGD + momentum
    # model.compile里的参数loss就是损失函数(目标函数)
    # print tower_0._keras_shape, tower_1._keras_shape
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # print model.summary()
    plot_model(model, to_file='ResNet_cifar10.png', show_shapes=True)

    begin_time = time.time()
    hinst = model.fit(X_train, Y_train, batch_size=model_batch_size, epochs=model_epochs, shuffle=True, verbose=1, validation_split=0.2)
    print '>> TEST'
    score = model.evaluate(X_test, Y_test, batch_size=model_batch_size, verbose=1)
    print "SCORE", score[0]
    print "AVERAGE ACCRUACY", score[1]
    end_time = time.time()
    print ">> WORKING TIME(s): %5.3f\n" % (end_time - begin_time)
    return


def ResNet50_ImageNet(X_train, Y_train, X_test, Y_test, input_shape, num_classes, model_batch_size, model_epochs, model_activation) :
    print '>> ResNet50 TRAINT AND TEST'
    size_1 = (1, 1)
    strides_1 = (1, 1)
    size_3 = (3, 3)
    strides_2 = (2, 2)

    init_shape = (3,224,224)
    g0_input = Input(shape=init_shape)

    # Group 1, conv+bn+act+pooling, output 56*56
    g1_padding1 = ZeroPadding2D((3, 3))(g0_input)
    g1_conv1    = Conv2D(64, (7, 7), strides=(2, 2), padding='valid')(g1_padding1)
    g1_bn1      = BatchNormalization(axis=1)(g1_conv1)
    g1_act1     = Activation(model_activation)(g1_bn1)
    g1_padding2 = ZeroPadding2D((1, 1))(g1_act1)
    g1_pooling1 = MaxPooling2D(pool_size=size_3, strides=strides_2)(g1_padding2)

    # Group 2, 1*ConvBlock + 2*IdentityBlock, output 56*56
    kernels_size = (64, 64, 256)
    kernels_strides = (1, 1)
    g2_conv_block1 = ConvBlock(g1_pooling1, kernels_size, kernels_strides, model_activation)
    g2_identity_block1 = IdentityBlock(g2_conv_block1, kernels_size, model_activation)
    g2_identity_block2 = IdentityBlock(g2_identity_block1, kernels_size, model_activation)

    # Group 3, 1*ConvBlock + 3*IdentityBlock, output 28*28
    kernels_size = (128, 128, 512)
    kernels_strides = (2, 2)
    g3_conv_block1 = ConvBlock(g2_identity_block2, kernels_size, kernels_strides, model_activation)
    g3_identity_block1 = IdentityBlock(g3_conv_block1, kernels_size, model_activation)
    g3_identity_block2 = IdentityBlock(g3_identity_block1, kernels_size, model_activation)
    g3_identity_block3 = IdentityBlock(g3_identity_block2, kernels_size, model_activation)

    # Group 4, 1*ConvBlock + 5*IdentityBlock, output 14*14
    kernels_size = (256, 256, 1024)
    kernels_strides = (2, 2)
    g4_conv_block1 = ConvBlock(g3_identity_block3, kernels_size, kernels_strides, model_activation)
    g4_identity_block1 = IdentityBlock(g4_conv_block1, kernels_size, model_activation)
    g4_identity_block2 = IdentityBlock(g4_identity_block1, kernels_size, model_activation)
    g4_identity_block3 = IdentityBlock(g4_identity_block2, kernels_size, model_activation)
    g4_identity_block4 = IdentityBlock(g4_identity_block3, kernels_size, model_activation)
    g4_identity_block5 = IdentityBlock(g4_identity_block4, kernels_size, model_activation)

    # Group 5, 1*ConvBlock + 2*IdentityBlock, output 7*7
    kernels_size = (512, 512, 2048)
    kernels_strides = (2, 2)
    g5_conv_block1     = ConvBlock(g4_identity_block5, kernels_size, kernels_strides, model_activation)
    g5_identity_block1 = IdentityBlock(g5_conv_block1, kernels_size, model_activation)
    g5_identity_block2 = IdentityBlock(g5_identity_block1, kernels_size, model_activation)

    # Group 6, avg pool, output 1*1
    g6_pooling1   = AveragePooling2D((7,7), strides=strides_1, padding='valid')(g5_identity_block2)

    # Group 7, avg pool, output 7*7
    g7_flatten = Flatten()(g6_pooling1)
    g7_output  = Dense(num_classes, kernel_initializer='normal', activation='softmax')(g7_flatten)

    model = Model(inputs=g0_input, outputs=[g7_output], name="ResNet50")

    # use SGD + momentum
    # model.compile里的参数loss就是损失函数(目标函数)
    # print tower_0._keras_shape, tower_1._keras_shape
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # print model.summary()
    plot_model(model, to_file='ResNet50.png', show_shapes=True)

    return

    begin_time = time.time()
    hinst = model.fit(X_train, Y_train, batch_size=model_batch_size, epochs=model_epochs, shuffle=True, verbose=1, validation_split=0.2)
    print '>> TEST'
    score = model.evaluate(X_test, Y_test, batch_size=model_batch_size, verbose=1)
    print "SCORE", score[0]
    print "AVERAGE ACCRUACY", score[1]
    end_time = time.time()
    print ">> WORKING TIME(s): %5.3f\n" % (end_time - begin_time)
    return

###################################################################

if __name__ == "__main__":
    #print os.environ['PATH']
    batch_size = 200
    epochs = 12
    activation = 'relu'
    # X_train, y_train, X_test, y_test = load_mnist_as_int()
    # # GNN(X_train, y_train, X_test, y_test, batch_size, epochs, 'tanh')

    # X_train, Y_train, X_test, Y_test, init_shape, num_classes = keras_data.load_mnist()
    X_train, Y_train, X_test, Y_test, init_shape, num_classes = keras_data.load_cifar10()

    ResNet_cifar10(X_train, Y_train, X_test, Y_test, init_shape, num_classes, batch_size, epochs, activation)
