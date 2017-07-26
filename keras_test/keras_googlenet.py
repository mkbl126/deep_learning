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

def GoogLeNetV1_obsolete(X_train, Y_train, X_test, Y_test, input_shape, num_classes, model_batch_size, model_epochs, model_activation) :
    print '>> GoogLeNetV1 TRAINT AND TEST'
    inception_size = (1, 1)
    inception_strides = (1, 1)
    model_pool_size = (3, 3)
    model_pool_strides = (2, 2)

    model = Graph()
    model.add_input(name="n00", input_shape=init_shape)

    # Layer 1, conv+pooling
    model.add_node(ZeroPadding2D((3,3)), name='n10', input='n00')
    model.add_node(Conv2D(64, (7, 7), strides=(2, 2), padding='valid', activation=model_activation), name='n11', input='n10')
    model.add_node(ZeroPadding2D((1,1)), name='n12', input='n11')
    model.add_node(MaxPooling2D(pool_size=model_pool_size, strides=model_pool_strides), name='n1', input='n12')

    # Layer 2, conv+pooling
    model.add_node(Conv2D(64, inception_size, strides=inception_strides, padding='valid', activation=model_activation), name='n20', input='n1')
    model.add_node(Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation=model_activation), name='n21', input='n20')
    model.add_node(ZeroPadding2D((1,1)), name='n22', input='n21')
    model.add_node(MaxPooling2D(pool_size=model_pool_size, strides=model_pool_strides), name='n2', input='n22')

    # Layer 3, conv+pooling, inception
    model.add_node(Conv2D(64, inception_size, strides=inception_strides, padding='valid', activation=model_activation), name='n300', input='n2')
    model.add_node(Conv2D(96, inception_size, strides=inception_strides, padding='valid', activation=model_activation), name='n310', input='n2')
    model.add_node(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=model_activation), name='n311', input='n310')
    model.add_node(Conv2D(16, inception_size, strides=inception_strides, padding='valid', activation=model_activation), name='n320', input='n2')
    model.add_node(Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation=model_activation), name='n321', input='n320')
    model.add_node(MaxPooling2D(pool_size=model_pool_size, strides=model_pool_strides, padding='same'), name='n330', input='n2')
    model.add_node(Conv2D(32, inception_size, strides=inception_strides, padding='valid', activation=model_activation), name='n331', input='n330')
    model.add_node(Merge.concatenate(inputs=['n300', 'n311', 'n321', 'n332']), name='n3')

    # Block 6, full
    model.add_node(Flatten())
    model.add_node(Dense(4096, kernel_initializer='normal', activation=model_activation), name='f1', input='n3')
    model.add_node(Dropout(0.5))

    # Block 8, full
    model.add_node(Dense(num_classes, kernel_initializer='normal', activation='softmax'), name='f2', input='f1')
    model.add_output(name='output1',input='f2')

    #使用SGD + momentum
    #model.compile里的参数loss就是损失函数(目标函数)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print model.summary()

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

def inceptionV1(input, kernels, model_activation) :
    inception_kernels = list(kernels)
    kernel_0 = inception_kernels[0]
    kernel_1_1 = inception_kernels[1]
    kernel_1_2 = inception_kernels[2]
    kernel_2_1 = inception_kernels[3]
    kernel_2_2 = inception_kernels[4]
    kernel_3 = inception_kernels[5]
    tower_0 = Conv2D(kernel_0, (1, 1), padding='same', activation=model_activation)(input)
    tower_1 = Conv2D(kernel_1_1, (1, 1), padding='same', activation=model_activation)(input)
    tower_1 = Conv2D(kernel_1_2, (3, 3), padding='same', activation=model_activation)(tower_1)
    tower_2 = Conv2D(kernel_2_1, (1, 1), padding='same', activation=model_activation)(input)
    tower_2 = Conv2D(kernel_2_2, (5, 5), padding='same', activation=model_activation)(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(kernel_3, (1, 1), padding='same', activation=model_activation)(tower_3)
    output = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=1)
    return output

def GoogLeNetV1(X_train, Y_train, X_test, Y_test, input_shape, num_classes, model_batch_size, model_epochs, model_activation) :
    print '>> GoogLeNetV1 TRAINT AND TEST'
    size_1 = (1, 1)
    strides_1 = (1, 1)
    size_3 = (3, 3)
    strides_2 = (2, 2)

    init_shape = (3,224,224)
    l0_input = Input(shape=init_shape)

    # Layer 1, conv+pooling, output 56*56
    l1_padding1 = ZeroPadding2D((3, 3))(l0_input)
    l2_conv1    = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', activation=model_activation)(l1_padding1)
    l1_padding2 = ZeroPadding2D((1, 1))(l2_conv1)
    l1_pooling1 = MaxPooling2D(pool_size=size_3, strides=strides_2, padding='valid')(l1_padding2)

    # Layer 2, conv+pooling, output 28*28
    l2_conv1    = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation=model_activation)(l1_pooling1)
    l2_conv2    = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation=model_activation)(l2_conv1)
    l2_padding1 = ZeroPadding2D((1,1))(l2_conv2)
    l2_pooling2 = MaxPooling2D(pool_size=size_3, strides=strides_2)(l2_padding1)

    # Layer 3, inception, output 28*28
    l3_inception_kernels = (64, 96, 128, 16, 32, 32)
    l3_inception1 = inceptionV1(l2_pooling2, l3_inception_kernels, model_activation)

    # Layer 4, inception, output 14*14
    l4_inception_kernels = (128, 128, 192, 32, 96, 64)
    l4_inception1 = inceptionV1(l3_inception1, l4_inception_kernels, model_activation)
    l4_padding1   = ZeroPadding2D((1,1))(l4_inception1)
    l4_pooling1   = MaxPooling2D(pool_size=size_3, strides=strides_2, padding='valid')(l4_padding1)

    # Layer 5, inception, output 14*14
    l5_inception_kernels = (192, 96, 208, 16, 48, 64)
    l5_inception1 = inceptionV1(l4_pooling1, l5_inception_kernels, model_activation)

    # Layer 6, inception, output 14*14
    l6_inception_kernels = (160, 112, 224, 24, 64, 64)
    l6_inception1 = inceptionV1(l5_inception1, l6_inception_kernels, model_activation)
    l6_pooling1   = AveragePooling2D((5,5), strides=(3,3), padding='valid')(l6_inception1)
    l6_conv1      = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation=model_activation)(l6_pooling1)
    l6_flatten    = Flatten()(l6_conv1)
    l6_dense1     = Dense(num_classes, kernel_initializer='normal', activation=model_activation)(l6_flatten)
    l6_output = l6_dense2 = Dense(num_classes, kernel_initializer='normal', activation='softmax')(l6_dense1)

    # Layer 7, inception, output 14*14
    l7_inception_kernels = (128, 128, 256, 24, 64, 64)
    l7_inception1 = inceptionV1(l6_inception1, l7_inception_kernels, model_activation)

    # Layer 8, inception, output 14*14
    l8_inception_kernels = (112, 144, 288, 32, 64, 64)
    l8_inception1 = inceptionV1(l7_inception1, l8_inception_kernels, model_activation)

    # Layer 9, inception, output 7*7
    l9_inception_kernels = (256, 160, 320, 32, 128, 128)
    l9_inception1 = inceptionV1(l8_inception1, l9_inception_kernels, model_activation)
    l9_padding1   = ZeroPadding2D((1,1))(l9_inception1)
    l9_pooling1   = MaxPooling2D(pool_size=size_3, strides=strides_2, padding='valid')(l9_padding1)
    l9_pooling2   = AveragePooling2D((5,5), strides=(3,3), padding='valid')(l9_inception1)
    l9_conv1      = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation=model_activation)(l9_pooling2)
    l9_flatten    = Flatten()(l9_conv1)
    l9_dense1     = Dense(num_classes, kernel_initializer='normal', activation=model_activation)(l9_flatten)
    l9_output = l9_dense2 = Dense(num_classes, kernel_initializer='normal', activation='softmax')(l9_dense1)

    # Layer 10, inception, output 7*7
    l10_inception_kernels = (256, 160, 320, 32, 128, 128)
    l10_inception1 = inceptionV1(l9_pooling1, l10_inception_kernels, model_activation)

    # Layer 11, inception
    l11_inception_kernels = (384, 192, 384, 48, 128, 128)
    l11_inception1 = inceptionV1(l10_inception1, l11_inception_kernels, model_activation)
    l11_pooling1   = AveragePooling2D((7,7), strides=strides_1, padding='valid')(l11_inception1)

    # Layer 12, inception
    l12_dropout1 = Dropout(0.4)(l11_pooling1)
    l12_flatten = Flatten()(l12_dropout1)
    l12_output = l12_dense1  = Dense(num_classes, kernel_initializer='normal', activation='softmax')(l12_flatten)

    model = Model(inputs=l0_input, outputs=[l6_output, l9_output, l12_output], name="GoogLeNetV1")

    #使用SGD + momentum
    #model.compile里的参数loss就是损失函数(目标函数)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print model.summary()
    plot_model(model, to_file='GoogLeNetV1.png', show_shapes=True)

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

def inceptionV3(input, kernels, model_activation) :
    inception_kernels = list(kernels)
    kernel_0 = inception_kernels[0]
    kernel_1_1 = inception_kernels[1]
    kernel_1_2 = inception_kernels[2]
    kernel_2_1 = inception_kernels[3]
    kernel_2_2 = inception_kernels[4]
    kernel_3 = inception_kernels[5]
    tower_0 = Conv2D(kernel_0, (1, 1), padding='same', activation=model_activation)(input)
    tower_1 = Conv2D(kernel_1_1, (1, 1), padding='same', activation=model_activation)(input)
    tower_1 = Conv2D(kernel_1_2, (1, 3), padding='same', activation=model_activation)(tower_1)
    tower_1 = Conv2D(kernel_1_2, (3, 1), padding='same', activation=model_activation)(tower_1)
    tower_2 = Conv2D(kernel_2_1, (1, 1), padding='same', activation=model_activation)(input)
    tower_2 = Conv2D(kernel_2_2, (1, 3), padding='same', activation=model_activation)(tower_2)
    tower_2 = Conv2D(kernel_2_2, (3, 1), padding='same', activation=model_activation)(tower_2)
    tower_2 = Conv2D(kernel_2_2, (1, 3), padding='same', activation=model_activation)(tower_2)
    tower_2 = Conv2D(kernel_2_2, (3, 1), padding='same', activation=model_activation)(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(kernel_3, (1, 1), padding='same', activation=model_activation)(tower_3)
    output = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=1)
    return output

def GoogLeNetV3(X_train, Y_train, X_test, Y_test, input_shape, num_classes, model_batch_size, model_epochs, model_activation) :
    print '>> GoogLeNetV3 TRAINT AND TEST'
    size_1 = (1, 1)
    strides_1 = (1, 1)
    size_3 = (3, 3)
    strides_2 = (2, 2)

    init_shape = (3,299,299)
    l0_input = Input(shape=init_shape)

    # Layer 1, conv+pooling, output 56*56
    l1_padding1 = ZeroPadding2D((3,3))(l0_input)
    l2_conv1    = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', activation=model_activation)(l1_padding1)
    l2_conv1    = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', activation=model_activation)(l1_padding1)


    l1_padding2 = ZeroPadding2D((1,1))(l2_conv1)
    l1_pooling1 = MaxPooling2D(pool_size=size_3, strides=strides_2, padding='valid')(l1_padding2)

    # Layer 2, conv+pooling, output 28*28
    l2_conv1    = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation=model_activation)(l1_pooling1)
    l2_conv2    = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation=model_activation)(l2_conv1)
    l2_padding1 = ZeroPadding2D((1,1))(l2_conv2)
    l2_pooling2 = MaxPooling2D(pool_size=size_3, strides=strides_2)(l2_padding1)

    # Layer 3, inception, output 28*28
    l3_inception_kernels = (64, 96, 128, 16, 32, 32)
    l3_inception1 = inceptionV3(l2_pooling2, l3_inception_kernels, model_activation)

    # Layer 4, inception, output 14*14
    l4_inception_kernels = (128, 128, 192, 32, 96, 64)
    l4_inception1 = inceptionV3(l3_inception1, l4_inception_kernels, model_activation)
    l4_padding1   = ZeroPadding2D((1,1))(l4_inception1)
    l4_pooling1   = MaxPooling2D(pool_size=size_3, strides=strides_2, padding='valid')(l4_padding1)

    # Layer 5, inception, output 14*14
    l5_inception_kernels = (192, 96, 208, 16, 48, 64)
    l5_inception1 = inceptionV3(l4_pooling1, l5_inception_kernels, model_activation)

    # Layer 6, inception, output 14*14
    l6_inception_kernels = (160, 112, 224, 24, 64, 64)
    l6_inception1 = inceptionV3(l5_inception1, l6_inception_kernels, model_activation)
    l6_pooling1   = AveragePooling2D((5,5), strides=(3,3), padding='valid')(l6_inception1)
    l6_conv1      = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation=model_activation)(l6_pooling1)
    l6_flatten    = Flatten()(l6_conv1)
    l6_dense1     = Dense(num_classes, kernel_initializer='normal', activation=model_activation)(l6_flatten)
    l6_output = l6_dense2 = Dense(num_classes, kernel_initializer='normal', activation='softmax')(l6_dense1)

    # Layer 7, inception, output 14*14
    l7_inception_kernels = (128, 128, 256, 24, 64, 64)
    l7_inception1 = inceptionV3(l6_inception1, l7_inception_kernels, model_activation)

    # Layer 8, inception, output 14*14
    l8_inception_kernels = (112, 144, 288, 32, 64, 64)
    l8_inception1 = inceptionV3(l7_inception1, l8_inception_kernels, model_activation)

    # Layer 9, inception, output 7*7
    l9_inception_kernels = (256, 160, 320, 32, 128, 128)
    l9_inception1 = inceptionV3(l8_inception1, l9_inception_kernels, model_activation)
    l9_padding1   = ZeroPadding2D((1,1))(l9_inception1)
    l9_pooling1   = MaxPooling2D(pool_size=size_3, strides=strides_2, padding='valid')(l9_padding1)
    l9_pooling2   = AveragePooling2D((5,5), strides=(3,3), padding='valid')(l9_inception1)
    l9_conv1      = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation=model_activation)(l9_pooling2)
    l9_flatten    = Flatten()(l9_conv1)
    l9_dense1     = Dense(num_classes, kernel_initializer='normal', activation=model_activation)(l9_flatten)
    l9_output = l9_dense2 = Dense(num_classes, kernel_initializer='normal', activation='softmax')(l9_dense1)

    # Layer 10, inception, output 7*7
    l10_inception_kernels = (256, 160, 320, 32, 128, 128)
    l10_inception1 = inceptionV3(l9_pooling1, l10_inception_kernels, model_activation)

    # Layer 11, inception
    l11_inception_kernels = (384, 192, 384, 48, 128, 128)
    l11_inception1 = inceptionV3(l10_inception1, l11_inception_kernels, model_activation)
    l11_pooling1   = AveragePooling2D((7,7), strides=strides_1, padding='valid')(l11_inception1)

    # Layer 12, inception
    l12_dropout1 = Dropout(0.4)(l11_pooling1)
    l12_flatten = Flatten()(l12_dropout1)
    l12_output = l12_dense1  = Dense(num_classes, kernel_initializer='normal', activation='softmax')(l12_flatten)

    model = Model(inputs=l0_input, outputs=[l6_output, l9_output, l12_output], name="GoogLeNetV1")
    # model = Model(inputs=l0_input, outputs=[l12_dropout1], name="GoogLeNetV1")

    #使用SGD + momentum
    #model.compile里的参数loss就是损失函数(目标函数)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print model.summary()
    plot_model(model, to_file='GoogLeNetV1.png', show_shapes=True)

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

    # CNN(X_train, y_train, X_test, y_test, init_shape, batch_size, epochs, activation)
    # LeNet5(X_train, Y_train, X_test, Y_test, init_shape, num_classes, batch_size, epochs, activation)
    # AlexNet(X_train, Y_train, X_test, Y_test, init_shape, num_classes, batch_size, epochs, activation)
    # VGG16(X_train, Y_train, X_test, Y_test, init_shape, num_classes, batch_size, epochs, activation)
    GoogLeNetV1(X_train, Y_train, X_test, Y_test, init_shape, num_classes, batch_size, epochs, activation)

"""
#使用data augmentation的方法
#一些参数和调用的方法，请看文档
datagen = ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False) # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(data)

for e in range(nb_epoch):
    print('-'*40)
    print('Epoch', e)
    print('-'*40)
    print("Training...")
    # batch train with realtime data augmentation
    progbar = generic_utils.Progbar(data.shape[0])
    for X_batch, Y_batch in datagen.flow(data, label):
        loss,accuracy = model.train(X_batch, Y_batch,accuracy=True)
        progbar.add(X_batch.shape[0], values=[("train loss", loss),("accuracy:", accuracy)] )

"""