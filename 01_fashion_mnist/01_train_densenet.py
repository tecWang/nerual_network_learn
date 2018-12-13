import os
import keras 
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

# tools
from result_visual import *
from tools import *

# model
# convolution block: = activation + conv2d + dropout 
def conv_block(x, nb_filter, dropout_rate=None, weight_decay=1e-4):
    # nb_filter = number of the convolution kernels
    print("nb_filter", nb_filter)
    x = Activation("relu")(x)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x

# transition block: = conv2d + dropout + avg_pool + bn
def transition_block(x, nb_filter, dropout_rate=None, weight_decay=1e-4):

    x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    
    return x

# build a dense_block where the output of each conv_block is fed to subsequent ones
# dense_block = convolution block + convolution block + ... + convolution block(the number of conv == nb_layers)
def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4):

    feature_list = [x]

    # concatnate multi conv 
    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=-1)(feature_list)
        nb_filter += growth_rate
    
    return x, nb_filter

# build final densenet
# final densenet model  =   Input + conv2d + nb \ 
#                           + (dense block + transition block) + ... + (dense block + transition block) \
#                           # dense block = (convolution block) + (convolution block) + ... + (convolution block)
#                           # convolution block: = activation + conv2d + dropout 
#                           # transition block = conv2d + dropout + avg_pool + bn
#                           + (dense_block) \
#                           + Activation + global_pool + dense(prediction)
def dense_net(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=32, 
                dropout_rate=None, weight_decay=1e-4, verbose=1):
    
    # first layer
    model_input = Input(shape=img_dim)
    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    nb_layers = int((depth-4)/3)

    # # initial conv2d
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(model_input)
    x = BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

    # # add dense blocks
    for block_index in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # add transition block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # # the last dense block does not has transition block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate, weight_decay=weight_decay)
    
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    pred = Dense(nb_classes, activation="softmax", kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    dense_net = Model(model_input, pred)
    # keras.models.Model()

    if verbose:
        print("Densenet -%d-%d created" % (depth, growth_rate))
    
    return dense_net


# variables
rows = 28
cols = 28
nb_classes = 10
img_dim = (rows, cols, 1)

depth = 13
nb_dense_block = 2

growth_rate = 32
nb_filter = 32

optimizer = "adadelta" # [adadelta, adam]
batch_size = 32
epochs = 20

train_round =   "d" + str(depth) + "-nd" + str(nb_dense_block) + \
                "-gr" +  str(growth_rate) + "-nf" + str(nb_filter) + \
                "-" + str(optimizer) + "-bs" + str(batch_size) + "-epo" + str(epochs)
result_path = "./result/densenet-result"
result_path = os.path.join(result_path, train_round)
mkdir(result_path)

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
print("x_train.shape", x_train.shape)
y_train = keras.utils.to_categorical(y_train).astype("float32")
y_test = keras.utils.to_categorical(y_test).astype("float32")


# build model
print("current model:", train_round)
model = dense_net(nb_classes, img_dim, depth, nb_dense_block=nb_dense_block, growth_rate=growth_rate)
# The Adam optimizer has a flexibel learning rate and is fast to solve the best answer
# The SGD optimizer has a stable learning rate, which means that the accuracy rate improvement effect is consistent from 0 to 1.

loss = keras.metrics.K.categorical_crossentropy
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
model_paint(model, filepath=result_path)

# fit model
history = model.fit(x_train, y_train, verbose=1, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

# save model and result
model.save(result_path + "/model.h5")
acc_loss_paint(history, filepath=result_path)