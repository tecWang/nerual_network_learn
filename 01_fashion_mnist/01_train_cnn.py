import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Model, Sequential
from keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt 
from result_visual import *
from tools import *

#####################################################################################
# author: tecwang
# email: tecwang@139.com
# desc: cnn-net by keras
#####################################################################################

#####################################################################################
# define model

def conv_layers(x, nb_filters=32, kernel_size=3, pool_size = 2, weight_decay=1e-4):
    x = layers.Conv2D(nb_filters, (kernel_size, kernel_size), padding="same", dilation_rate=2)(x)
    # x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization(axis=-1, beta_regularizer=l2(weight_decay))(x)
    x = layers.MaxPooling2D((pool_size, pool_size))(x)
    x = layers.Dropout(0.2)(x)
    return x

def dense_layers(x, units, weight_decay=1e-4):
    x = layers.Dense(units)(x)
    # x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization(axis=-1, beta_regularizer=l2(weight_decay))(x)
    x = layers.Dropout(0.2)(x)
    return x  

def cnn_model(rows, cols, channels=1, conv_units=[32, 64], dense_units=[512, 128, 10], weight_decay=1e-4):

    model_input = layers.Input(shape=(rows, cols, channels))
    x = model_input

    # conv period
    for i in range(len(conv_units)):
        x = conv_layers(x, nb_filters=conv_units[i], kernel_size=3)
    # max pool    
    x = layers.MaxPooling2D((2, 2))(x)  # do not need activation, so do not need dropout layer

    # dense period
    x = layers.Flatten()(x) # do not need activation, so do not need dropout layer
    for j in range(len(dense_units)-1):
        x = dense_layers(x, dense_units[j])

    # prediction
    prediction = layers.Dense(dense_units[-1], activation="softmax")(x)
    model = keras.models.Model(inputs=model_input, outputs=prediction)
    
    return model

#####################################################################################
# define hyper-params

rows = 28
cols = 28
channels = 1

nb_conv = 2
conv_units = [64, 128]
dense_units = [128, 10]
batch_size = 128
epochs = 20
loss = "categorical_crossentropy" # [categorical_crossentropy, mse]
optimizer = "adam"  # [adadelta, adam]

# folder where the training result are saved
train_round =   "cunits" + str(conv_units) + "-dunits" + str(dense_units) + \
                "-bs" + str(batch_size) + "-epo" + str(epochs) + \
                "-[maxpool,relu*2,bn*2,hidden-drop]"
                # "-" + str(loss) + "-" + str(optimizer) + \

print('#####################################################################################')
print("result is stored in ", train_round)
print('#####################################################################################')

result_path = "./result/cnn-result/"
result_path = os.path.join(result_path, train_round)
mkdir(result_path)

#####################################################################################
# load fashion mnist datasets

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# onehot label
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onthot = keras.utils.to_categorical(test_labels)

# convert 0-255 to 0-1 and convert to floating point
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images_more_dim = train_images.reshape(-1, 28, 28,1).astype('float32')  
test_images_more_dim = test_images.reshape(-1,28, 28,1).astype('float32')
print("train_images.shape", train_images.shape)
print("train_labels", train_labels)     # [9 0 0 ... 3 0 5], corresponds to the contents of the class_names array

#####################################################################################
# make model

model = cnn_model(rows, cols, channels, conv_units=conv_units, dense_units=dense_units)
model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
print(model.summary())
model_paint(model, filepath=result_path)

history = model.fit(train_images_more_dim, train_labels_onehot, 
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(test_images_more_dim, test_labels_onthot),
                    callbacks=[
                        keras.callbacks.TensorBoard(
                            log_dir=result_path + '/logs', histogram_freq=0, batch_size=32, write_graph=True, 
                            write_grads=True, write_images=True),
                        keras.callbacks.CSVLogger(result_path + "/training.log", separator='\t', append=False)])
# model.save(result_path +"/fashion_mnist_model.h5")


#####################################################################################
# # evaluate model
test_loss, test_acc = model.evaluate(test_images_more_dim, test_labels_onthot)
print('Test accuracy:', test_acc)

# result visual
acc_loss_paint(history, filepath=result_path)