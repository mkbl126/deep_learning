# coding=utf-8

import os
import numpy as np
from PIL import Image

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras import backend as K
from keras import utils

###################################################################

def load_cifar100() :
    # The data, shuffled and split between train and test sets:
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    init_shape = ( X_train.shape[-3],  X_train.shape[-2],  X_train.shape[-1])
    # Convert class vectors to binary class matrices.
    num_classes = 100
    Y_train = utils.to_categorical(Y_train, num_classes)
    Y_test = utils.to_categorical(Y_test, num_classes)
    print(">> LOADED cifar10 -> X_train(%d), X_test(%d), Y_train(%d), Y_test(%d) Shape%s" % (len(X_train), len(X_test), len(Y_train), len(Y_test), init_shape))
    return X_train, Y_train, X_test, Y_test, init_shape, num_classes

def load_cifar10() :
    # The data, shuffled and split between train and test sets:
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    init_shape = ( X_train.shape[-3],  X_train.shape[-2],  X_train.shape[-1])
    # Convert class vectors to binary class matrices.
    num_classes = 10
    Y_train = utils.to_categorical(Y_train, num_classes)
    Y_test = utils.to_categorical(Y_test, num_classes)
    print(">> LOADED cifar10 -> X_train(%d), X_test(%d), Y_train(%d), Y_test(%d) Shape%s" % (len(X_train), len(X_test), len(Y_train), len(Y_test), init_shape))
    return X_train, Y_train, X_test, Y_test, init_shape, num_classes

def load_mnist() :
    # the data, shuffled and split between train and test sets
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # reshape data into 28*28=784
    img_rows = 28
    img_cols = 28
    print(X_train.shape)
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        init_shape = (X_train.shape[-3],  X_train.shape[-2],  X_train.shape[-1])
        # init_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        init_shape = ( X_train.shape[-2],  X_train.shape[-1],  X_train.shape[-3])
        # init_shape = (img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # convert class vectors to binary class matrices
    num_classes = 10
    Y_train = utils.to_categorical(Y_train, num_classes)
    Y_test = utils.to_categorical(Y_test, num_classes)
    print(">> LOADED mnist -> X_train(%d), X_test(%d), Y_train(%d), Y_test(%d) Shape%s" % (len(X_train), len(X_test), len(Y_train), len(Y_test), init_shape))
    return X_train, Y_train, X_test, Y_test, init_shape, num_classes

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
    print(">> LOADED DATA -> X_train(%d), Y_train(%d), X_test(%d), Y_test(%d)" % (len(X_train), len(Y_train), len(X_test), len(Y_test)))
    return X_train, Y_train, X_test, Y_test
    
def load_minist_image():
    # read images in mnist folder, the size is 1*28*28 for grayscal
    # for color image change 1 to 3��data[i,:,:,:] = arr to data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
    folder = r"C:\DL\mnist"
    imgs = os.listdir(folder)
    num = len(imgs)
    #num = min(1000, len(imgs))
    data = np.empty((num,1,28,28),dtype="float32")
    label = np.empty((num,),dtype="uint8")
    for i in range(num):
        img = Image.open(os.path.join(folder,imgs[i]))
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
        data /= np.max(data)
        data -= np.mean(data)
        if i % 100==0 :
            print(i)
        if i>100 :
            break
    return data,label