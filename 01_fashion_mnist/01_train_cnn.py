import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Model, Sequential

import numpy as np
import matplotlib.pyplot as plt 
from result_visual import *

result_path = "./result/cnn-result/"

# load fashion mnist datasets
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onthot = keras.utils.to_categorical(test_labels)
train_images_more_dim = train_images.reshape(-1, 28, 28,1).astype('float32')  
test_images_more_dim = test_images.reshape(-1,28, 28,1).astype('float32')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("train_images.shape", train_images.shape)
print("train_labels", train_labels)     # [9 0 0 ... 3 0 5], corresponds to the contents of the class_names array


# preview first train image
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.show()
# plt.savefig("./[Image]first_image_[0-255].png")

# convert 0-255 to 0-1 and convert to floating point
train_images = train_images / 255.0
test_images = test_images / 255.0

# display the first 25 pictures of the training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)  # num must be 1 <= num <= 25, not 0
    plt.xticks([])
    plt.yticks([])
    # plt.grid(False)
    # plt.imshow(train_images[i])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig(result_path + "first_25_training_images.png")


# build nn model
model = Sequential()
model.add(layers.Conv2D(32, (3, 3),
                 input_shape=(28, 28, 1), 
                 activation="relu",
                 padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.35))    
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss=keras.metrics.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

    
model.summary()
history = model.fit(train_images_more_dim, train_labels_onehot, 
    batch_size=128, validation_split=0.2, epochs=20, verbose=1)
model.save(result_path + "fashion_mnist_model.h5")

# evaluate model
test_loss, test_acc = model.evaluate(test_images_more_dim, test_labels_onthot)
print('Test accuracy:', test_acc)

# result visual
acc_loss_paint(history, filepath=result_path + "acc_loss.png")
model_paint(model, filepath=result_path + "model.png")