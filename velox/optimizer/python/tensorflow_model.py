import time
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import layers

input_dimension = 597540
batch = 1000

# create the structure of the model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(input_dimension,), batch_size=batch))
model.add(layers.Dense(1024, activation= 'relu'))
model.add(layers.Dense(14588, activation= 'softmax'))

# set the weights and bias of the model using float precision
for layer in model.layers:
    print (layer.get_weights()[0].shape)
    a,b = layer.get_weights()[0].shape
    w = tf.dtypes.cast(tf.random.normal([a,b], stddev = 2, mean = 0, seed =1), tf.float32)
    b = tf.dtypes.cast(tf.random.normal((layer.get_weights()[1].shape), stddev = 2, mean = 0, seed =1), tf.float32)
    layer.set_weights([w, b])

# compile the model
model.compile()

# # save the model
model.save('amazon_14K.h5')