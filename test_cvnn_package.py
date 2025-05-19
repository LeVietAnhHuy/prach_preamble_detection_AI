import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from cvnn import layers
import numpy as np

tfds.disable_progress_bar()
tf.enable_v2_behavior()


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train_test'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([  # Remember to cast the dtype to float32
    layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.float32),
    layers.ComplexDense(128, activation='cart_relu', dtype=np.float32),
    layers.ComplexDense(10, activation='softmax_real', dtype=np.float32)
])
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'],
              )
model.fit(ds_train, epochs=6, validation_data=ds_test, verbose=verbose, shuffle=False)
