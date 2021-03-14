import os
import matplotlib.pyplot 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds
from tensorflow_datasets import text
#from matplotlib.pyplot import text

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# (ds_train, ds_test), ds_info = tfds.load(
#     "mnist", 
#     split = ["train", "test"],
#     shuffle_files = True,
#     as_supervised = True, #(img, label)
#     with_info = True,
# )

# # fig = tfds.show_examples(ds_train, ds_info, rows=4, cols=4)
# # print(ds_info)

# def normalize_img(image, label):
#     # normalize image
#     return tf.cast(image, tf.float32)/255.0, label

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# BATCH_SIZE = 64
# ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
# ds_train = ds_train.batch(BATCH_SIZE)
# ds_train = ds_train.prefetch(AUTOTUNE)

# ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.prefetch(AUTOTUNE)

# model = keras.Sequential([
#     keras.Input((28, 28, 1)),
#     layers.Conv2D(32, 3, activation='relu'),
#     layers.Flatten(),
#     layers.Dense(10),
# ])

# model.compile(
#     optimizer=keras.optimizers.Adam(lr=0.001),
#     loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics = "accuracy",
# )

# model.fit(ds_train, epochs=5, verbose=2)
# model.evaluate(ds_test)

#This movie was shit -> 0
(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    shuffle_files = True,
    as_supervised = True,
    with_info = True,
    )

print(ds_info)

for text. label in ds_train:
    print(text)
    import sys
    sys.exit()
#This movie is the shit -> 1

