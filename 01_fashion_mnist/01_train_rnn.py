import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

# tool
import matplotlib.pyplot as plt

# variables
batch_size = 128
epochs = 20
num_classes = 10

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# convert [0-255] to [0-1]
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
x_train /= 255.0
x_test /= 255.0
print(x_train.shape)

# one hot encode
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)



# define model
x = Input(shape=(28, 28, 1))
row_hidden = 128
col_hidden = 128
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
encoded_cols = LSTM(col_hidden)(encoded_rows)
outputs = Dense(num_classes, activation="softmax")(encoded_cols)
model = Model(x, outputs)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())


# fit model
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))


# Evaluation.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])