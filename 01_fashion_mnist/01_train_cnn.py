import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Model, Sequential
from keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt 
from tools import *

#####################################################################################
# author: tecwang
# email: tecwang@139.com
# desc: cnn-net by keras
#####################################################################################



#####################################################################################
# dense-net model

def conv_block(x, nb_filter, dropout_rate=None, weight_decay=1e-4):
    # nb_filter = number of the convolution kernels
    x = layers.Conv2D(nb_filter, (3, 3),
            kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay), 
            padding="same", use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)
    return x

def transition_block(x, nb_filter, dropout_rate=None, weight_decay=1e-4):
    # transition block: = conv2d + dropout + avg_pool + bn

    x = layers.Conv2D(nb_filter, (1, 1),
            kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay), 
            padding="same", use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4):
    # dense_block = convolution block + convolution block + ... + convolution block(the number of conv == nb_layers)

    feature_list = [x]

    # concatnate multi conv 
    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = layers.Concatenate(axis=-1)(feature_list)
        nb_filter += growth_rate
    
    return x, nb_filter

def dense_net(img_dim, nb_classes, 
                depth=13, nb_dense_block=2, nb_filter=32, growth_rate=12,  
                dropout_rate=None, weight_decay=1e-4):

    # first layer
    model_input = layers.Input(shape=img_dim)
    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    nb_layers = int((depth-4)/3)

    # # initial conv2d
    x = layers.Conv2D(nb_filter, (3, 3), 
        kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay),
        padding="same", use_bias=False)(model_input)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # # add dense blocks
    for block_index in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # add transition block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # # the last dense block does not has transition block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate, weight_decay=weight_decay)
    
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)
    pred = layers.Dense(nb_classes, activation="softmax", kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    dense_net = Model(model_input, pred)
    
    return dense_net

#####################################################################################
# cnn model

def conv_layers(x, nb_filters=32, kernel_size=3, pool_size = 2, weight_decay=1e-4, dropout_rate=None):
    x = layers.Conv2D(nb_filters, (kernel_size, kernel_size), activation="relu",
                        use_bias=False, padding="same", dilation_rate=2, 
                        kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    # x = layers.MaxPooling2D((pool_size, pool_size))(x)
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    return x

def dense_layers(x, units, weight_decay=1e-4, dropout_rate=None):
    x = layers.Dense(units, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    return x  

def cnn_net(img_dim, conv_units=[32, 64], dense_units=[128, 10], weight_decay=1e-4, dropout_rate=None):

    model_input = layers.Input(shape=img_dim)
    x = model_input

    # conv period
    for i in range(len(conv_units)):
        x = conv_layers(x, nb_filters=conv_units[i], kernel_size=3)
    x = layers.MaxPooling2D((2, 2))(x)  # do not need activation, so do not need dropout layer

    # dense period
    x = layers.Flatten()(x) # do not need activation, so do not need dropout layer
    for j in range(len(dense_units)-1):
        x = dense_layers(x, dense_units[j])
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # prediction
    prediction = layers.Dense(dense_units[-1], activation="softmax", 
            bias_regularizer=l2(weight_decay), kernel_regularizer=l2(weight_decay))(x)
    model = keras.models.Model(inputs=model_input, outputs=prediction)
    
    return model

#####################################################################################
# define hyper-params

img_dim = (28, 28, 1)
nb_classes = 10
nb_conv = 2
conv_units = [32, 64, 128]
dense_units = [128, nb_classes]
dropout_rate = 0.2
batch_size = 128
epochs = 2

depth=19            # (depth - 4) % 3 == 0, depth = 4, 7, 10, ...
nb_dense_block=3    # nb_dense_block = 1, 2, 3, 4
nb_filter=32        
growth_rate=32 
weight_decay=1e-4

loss = "categorical_crossentropy" # [categorical_crossentropy, mse]
optimizer = "adam"  # [adadelta, adam]

#####################################################################################
# folder where the training result are saved

# train_round =   "cunits" + str(conv_units) + "-dunits" + str(dense_units) + \
#                 "-dr" + str(dropout_rate) + \
#                 "-bs" + str(batch_size) + "-epo" + str(epochs) + \
#                 "-[maxpool,relu*2,bn*2,hidden-drop,kernel-regular]"
#                 # "-" + str(loss) + "-" + str(optimizer) + \

train_round =   "d" + str(depth) + "-nd" + str(nb_dense_block) + \
                "-nf" + str(nb_filter) + "-gr" + str(growth_rate) + \
                "-bs" + str(batch_size) + "-epo" + str(epochs) + \
                "-[testdata]"

result_path = "./result/"
result_path = os.path.join(result_path, train_round)
mkdir(result_path)

print('=====================================================================================')
print('')
print("result is stored in ", train_round)
print('')
print('=====================================================================================')

#####################################################################################
# load fashion mnist datasets

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()

# onehot label
train_label_onehot = keras.utils.to_categorical(train_label)
test_label_onehot = keras.utils.to_categorical(test_label)

# convert 0-255 to 0-1 and convert to floating point
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images_more_dim = train_images.reshape(-1, 28, 28,1).astype('float32')  
test_images_more_dim = test_images.reshape(-1,28, 28,1).astype('float32')

# Test set
test_set = test_images_more_dim[0: int(test_images_more_dim.shape[0]*0.8)]
test_label = test_label_onehot[0: int(test_images_more_dim.shape[0]*0.8)]
# Verification set
verification_set = test_images_more_dim[int(test_images_more_dim.shape[0]*0.8):]
verification_label = test_label_onehot[int(test_images_more_dim.shape[0]*0.8):]

print("train_images.shape", train_images.shape)
print("train_label", train_label)     # [9 0 0 ... 3 0 5], corresponds to the contents of the class_names array

#####################################################################################
# make model

# model = cnn_net(img_dim, conv_units=conv_units, dense_units=dense_units, dropout_rate=0.2)
model = dense_net(img_dim, nb_classes, depth=depth, nb_dense_block=nb_dense_block, nb_filter=nb_filter,
                    growth_rate=growth_rate, dropout_rate=dropout_rate, weight_decay=1e-4)

model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
# print(model.summary())
keras.utils.plot_model(model, result_path + "/model.png", show_shapes=True, show_layer_names=True, rankdir="TB")


history = model.fit(train_images_more_dim, train_label_onehot, 
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(test_set, test_label),
                    callbacks=[
                        keras.callbacks.TensorBoard(
                            log_dir=result_path + '/logs', histogram_freq=0, batch_size=32, write_graph=True, 
                            write_grads=True, write_images=True),
                        keras.callbacks.CSVLogger(result_path + "/training.log", separator='\t', append=False)])
model.save(result_path +"/fashion_mnist_model.h5")


#####################################################################################
# # evaluate model
test_loss, test_acc = model.evaluate(verification_set, verification_label)
print('Test accuracy:', test_acc)

optimizer = "sgd"
print('=====================================================================================')
print('')
print("change optimizer to %s and continue training", optimizer)
print('')
print('=====================================================================================')

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
history = model.fit(train_images_more_dim, train_label_onehot, 
                    batch_size=batch_size, epochs=epochs*5, verbose=1,
                    validation_data=(test_set, test_label),
                    callbacks=[
                        keras.callbacks.TensorBoard(
                            log_dir=result_path + '/logs_sgd', histogram_freq=0, batch_size=32, write_graph=True, 
                            write_grads=True, write_images=True),
                        keras.callbacks.CSVLogger(result_path + "/training_sgd.log", separator='\t', append=False)])
model.save(result_path +"/model_dgd.h5")
