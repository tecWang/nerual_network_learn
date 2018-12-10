import keras
import tensorflow as tf
from keras import models
from keras.models import load_model
import matplotlib.pyplot as plt

# load trained model
model = load_model("./fashion_mnist_model.h5")
print(model.summary())

# load fashion mnist datasets
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
test_labels_onthot = keras.utils.to_categorical(test_labels)
test_images = test_images.astype('float32')
test_image = test_images[0].reshape(-1, 28, 28, 1)
print(test_image.shape)

# preview first image
plt.figure()
plt.imshow(train_images[0])
# plt.show()

# extracts the outputs of each layers
layers_outputs = [layer.output for layer in model.layers[:]]
print(layers_outputs)

# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layers_outputs)

# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(test_image)

for k in range(4):
    plt.figure()
    print(activations[k].shape)
    for i in range(activations[k].shape[3]):
        plt.subplot(8, activations[k].shape[3]/8, i+1)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(activations[k][0, :, :, i])
    plt.savefig("./[Image]activations_layers_" + str(k+1) + ".png")
