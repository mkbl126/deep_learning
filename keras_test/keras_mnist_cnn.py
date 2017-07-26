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

import os
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"

import time
import random
import numpy as np
#
# from __future__ import absolute_import
# from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

#from six.moves import range

from keras.datasets import mnist

import keras

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K




###################################################################


def load_mnist() :
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    # reshape data into 28*28=784
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    # if K.image_data_format() == 'channels_first':
    #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_train = X_train.astype('float32')
    X_train /= 255
    X_test /= 255

    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    num_classes = 10
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)


    print ">> LOADED DATA -> X_train(%d), Y_train(%d), X_test(%d), Y_test(%d)" % (len(X_train), len(Y_train), len(X_test), len(Y_test))
    return X_train, Y_train, X_test, Y_test


def load_mnist_as_int() :
    # load mnist data from net
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape data into 28*28=784
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    # X_train = X_train / 255
    # X_test = X_test / 255

    # transfor index into one-hot matrix
    Y_train = (np.arange(10) == y_train[:, None]).astype(int)
    Y_test = (np.arange(10) == y_test[:, None]).astype(int)
    print ">> LOADED DATA -> X_train(%d), Y_train(%d), X_test(%d), Y_test(%d)" % (len(X_train), len(Y_train), len(X_test), len(Y_test))
    return X_train, Y_train, X_test, Y_test

###################################################################

def GNN(X_train, Y_train, X_test, Y_test, model_batch_size, model_epochs, model_activation) :
    print '>> GNN TRAINT AND TEST'
    model = Sequential()
    # input layer 28*28=784
    model.add(Dense(500, kernel_initializer='glorot_uniform', input_dim=784))
    model.add(Activation(model_activation))
    model.add(Dropout(0.5))
    # hidden layer=500
    model.add(Dense(500, kernel_initializer='glorot_uniform', input_dim=500))
    model.add(Activation(model_activation))
    model.add(Dropout(0.5))
    # output layer=10
    model.add(Dense(10, kernel_initializer='glorot_uniform', input_dim=500))
    model.add(Activation('softmax'))

    # set learning rate, loss function=cross_entropy
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # 开始训练，batch_size就是batch_size，epochs就是最多迭代的次数， shuffle就是是否把数据随机打乱之后再进行训练
    # verbose是屏显模式，官方这么说的：verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
    # 就是说0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据
    # show_accuracy就是显示每次迭代后的正确率
    # validation_split就是拿出百分之多少用来做交叉验证

    begin_time = time.time()
    model.fit(X_train, Y_train, batch_size=model_batch_size, epochs=model_epochs, shuffle=True, verbose=1, validation_split=0.3)
    print '>> TEST'
    score = model.evaluate(X_test, Y_test, batch_size=model_batch_size, verbose=1)
    print "SCORE", score[0]
    print "AVERAGE ACCRUACY", score[1]
    end_time = time.time()
    print "\n>> WORKING TIME(s): %5.3f " % (end_time - begin_time)

###################################################################

def CNN(X_train, Y_train, X_test, Y_test, model_batch_size, model_epochs, model_activation) :
    print '>> CNN TRAINT AND TEST'
    model = Sequential()

    #第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
    #border_mode可以是valid或者full，具体看这里说明：http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
    #激活函数用tanh
    #你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
    model.add(Convolution2D(16, (5, 5), padding='valid', input_shape=(1,28,28)))
    model.add(Activation(model_activation))

    #第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
    #激活函数用tanh, 采用maxpooling，poolsize为(2,2)
    model.add(Convolution2D(32, (3, 3), padding='valid'))
    model.add(Activation(model_activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #第三个卷积层，16个卷积核，每个卷积核大小3*3
    #激活函数用tanh, 采用maxpooling，poolsize为(2,2)
    model.add(Convolution2D(16, (3, 3), padding='valid'))
    model.add(Activation(model_activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    #全连接层，先将前一层输出的二维特征图flatten为一维的。
    #Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4
    #全连接有128个神经元节点,初始化方式为normal
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='normal'))
    model.add(Activation(model_activation))
    #model.add(Dropout(0.5))

    #Softmax分类，输出是10类别
    model.add(Dense(10, kernel_initializer='normal'))
    model.add(Activation('softmax'))

    #使用SGD + momentum
    #model.compile里的参数loss就是损失函数(目标函数)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
    #数据经过随机打乱shuffle=True。verbose=1，训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
    #validation_split=0.2，将20%的数据作为验证集
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test  = X_test.reshape(X_test.shape[0], 1, 28, 28)
    begin_time = time.time()
    hinst = model.fit(X_train, Y_train, batch_size=model_batch_size, epochs=model_epochs, shuffle=True, verbose=1, validation_split=0.2)
    print '>> TEST'
    score = model.evaluate(X_test, Y_test, batch_size=model_batch_size, verbose=1)
    print "SCORE", score[0]
    print "AVERAGE ACCRUACY", score[1]
    end_time = time.time()
    print ">> WORKING TIME(s): %5.3f\n" % (end_time - begin_time)

###################################################################

def AlexNet(X_train, Y_train, X_test, Y_test, model_batch_size, model_epochs, model_activation) :
    print '>> AlexNet TRAINT AND TEST'
    model = Sequential()
    return

###################################################################


if __name__ == "__main__":
    #print os.environ['PATH']
    batch_size = 200
    epochs = 12
    # X_train, y_train, X_test, y_test = load_mnist_as_int()
    # GNN(X_train, y_train, X_test, y_test, batch_size, epochs, 'tanh')
    # CNN(X_train, y_train, X_test, y_test, batch_size, epochs, 'relu')
    load_mnist()


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