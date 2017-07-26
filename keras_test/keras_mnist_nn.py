# coding=utf-8
import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

if __name__ == "__main__":
    # load mnist data from net
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape data into 28*28=784
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    # X_train = X_train / 255
    # X_test = X_test / 255
    # print X_train[0]

    # transfor index into one-hot matrix
    Y_train = (np.arange(10) == y_train[:, None]).astype(int)
    Y_test = (np.arange(10) == y_test[:, None]).astype(int)

    # print y_train, len(y_train), len(y_train[:, ]), y_train[:, ]
    # print np.arange(10), np.arange(10) == y_train[1]
    # print y_train, Y_train

    model = Sequential()
    # input layer 28*28=784
    model.add(Dense(500, kernel_initializer='glorot_uniform', input_dim=784))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    # hidden layer=500
    model.add(Dense(500, kernel_initializer='glorot_uniform', input_dim=500))
    model.add(Activation('sigmoid'))
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
    model.fit(X_train, Y_train, batch_size=200, epochs=10, shuffle=True, verbose=1, validation_split=0.3)
    print '>> test set'
    score = model.evaluate(X_test, Y_test, batch_size=200, verbose=1)
    print "", score[0]
    print "average accuracy", score[1]

    end_time = time.time()
    print "\n>> WORKING TIME(s): %5.3f " % (end_time - begin_time)
