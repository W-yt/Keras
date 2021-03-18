import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,Activation,MaxPool2D,Flatten,Dense
from keras.optimizers import Adam

nb_class = 10
nb_epoch = 4
batchsize = 128

#prepare your data mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#setup data shape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255.0
x_test /= 255.0

#one-hot[0,0,0,0,0,0,1,0,0,0](一个是热的)
y_train = np_utils.to_categorical(y_train,nb_class)
y_test = np_utils.to_categorical(y_test,nb_class)

#setup model
model = Sequential()

#1st conv2Dlayer
model.add(Convolution2D(filters = 32, # 过滤器走32次
                        kernel_size = (5,5), # 过滤器尺寸
                        padding = "same", # padding是处理过滤器走的时候最后一次不足的地方的
                        input_shape = (28,28,1)
))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (2,2), #
                    strides = (2,2), #步长
                    padding = "same"
))

#2nd conv2D layer
model.add(Convolution2D(filters = 64,
                        kernel_size = (5,5),
                        padding = "same",
                        input_shape = (28,28,1)
))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (2,2),
                    strides = (2,2),
                    padding = "same"
))

#1st Fully connected Dense
model.add(Flatten()) #降维
model.add(Dense(1024))
model.add(Activation("relu"))

#2nd Fully connected Dense
model.add(Dense(nb_class))
model.add(Activation("softmax"))

#define optimizer and setup para
adam = Adam(lr = 0.001) # learning rate

#compile
model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

#run/fireup network
model.fit(x_train,
          y_train,
          epochs = nb_epoch,
          batch_size = batchsize,
          validation_data = (x_test,y_test),
          verbose = 1)




