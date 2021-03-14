import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np 
import random

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#plt.imshow(x_train[0], cmap="gray")
# print(x_train[0].shape)

#print(x_train[0])

x_train = x_train / 255.0
x_test = x_test / 255.0

encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = keras.layers.Flatten()(encoder_input)
encoder_output = keras.layers.Dense(64, activation="relu")(x)

encoder = keras.Model(encoder_input, encoder_output, name = "encoder")

decoder_input = keras.layers.Dense(64, activation="relu")(encoder_output)
x = keras.layers.Dense(784, activation="relu")(decoder_input)
decoder_output = keras.layers.Reshape((28, 28, 1))(x)
opt = keras.optimizers.Adam(lr=0.001)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
autoencoder.compile(opt, loss="mse")
autoencoder.fit(x_train, x_train, epochs=3, batch_size=32, validation_split=0.1)

example = encoder.predict([x_test[0].reshape(-1, 28, 28, 1)])[0]
print(example.shape)

plt.imshow(example.reshape(8, 8), cmap="gray")

plt.imshow(x_test[0], cmap="gray")


ae_out = autoencoder.predict([x_test[0].reshape(-1, 28, 28, 1)])[0]
img = ae_out[0] 
plt.imshow(ae_out, cmap="gray")
plt.show()

def add_noise(img, random_chance=5):
    noisy = []
    for row in img:
        new_row = []
        for pix in row:
            if random.choice(range(100)) <= random_chance:
                new_val = random.uniform(0, 1)
                new_row.append(new_val)
            else:
                new_row.append(pix)
            noisy.append(new_row)
        return np.array(noisy)

noisy = add_noise(x_test[0])
plt.imshow(noisy, cmap="gray")






