import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split = ["train", "test"],
    shuffle_files = True, 
    as_supervised = True, 
    with_info = True,
)

def normalize_img(image, label):
    """Normalize images"""
    return tf.cast(image, tf.float32) / 255.0, label

def augment(image, label):
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

#Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

#Setup for test dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)

class_names = [
    "Airplane", 
    "Automobile", 
    "Bird", 
    "Cat",
    "Deer",
    "Dog", 
    "Frog", 
    "Horse",
    "Ship", 
    "Truck",
]

def get_model():
    model = keras.Sequential(
        [
            layers.Input((32, 32, 3)),
            layers.Conv2D(8, 3, padding="same", activation="relu"), 
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(), 
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1), 
            layers.Dense(10),
        ]
    )

    return model

model = get_model()

model.compile(
    optimizer = keras.optimizers.Adam(lr=0.001),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ["accuracy"],
)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir = "tb_callback_dir", histogram_freq=1, 

)

model.fit(
    ds_train, 
    epochs=5, 
    validation_data=ds_test, 
    callbacks = [tensorboard_callback],
    verbose = 2,
)


num_epochs = 1
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(lr=0.001)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
train_writer = tf.summary.create_file_writer("logs/train/")
test_writer = tf.summary.create_file_writer("logs/test/")
train_step = test_step = 0

for epoch in range(num_epochs):
    #Iterate through training step
    for batch_idx, (x, y) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=False)
            loss = loss_fn(y, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y, y_pred)

    with train_writer.as_default():
        tf.summary.scalar("Loss", loss)
        tf.summary.scalar(
            "Accuracy", acc_metric.result(), step=epoch,
        )

    #Reset accuracy in between epochs (and for testing and test)
    acc_metric.reset_states()

    #Iterate through test set
    for batch_idx in enumerate(ds_test):
        y_pred = model(x, training=False)
        loss = loss_fn(y, y_pred)
        acc_metric.update_state(y, y_pred)

    with train_writer.as_default():
        tf.summary.scalar("Loss", loss)
        tf.summary.scalar(
            "Accuracy", acc_metric.result(), step=epoch,
        )


    acc_metric.reset_states()

