import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt # plotting library


from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout
from keras.optimizers import Adam ,RMSprop
from keras import  backend as K
from keras.datasets import mnist

from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model

import time

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(1)
# load dataset

batch_size = 128
hidden_units = 1024
dropout = 0.45
model = load_model('m1024')
config = tf.config.threading.get_inter_op_parallelism_threads()
num_samples = 1000

# Print the number of threads being used
print("Number of inter-op parallelism threads:", config)
config = tf.config.threading.get_intra_op_parallelism_threads()

# Print the number of threads being used
print("Number of intra-op parallelism threads:", config)

start_time = time.time()
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train[:num_samples]
#y_train = y_train[:num_samples]
print("TensorFlow version:", tf.__version__)

# count the number of unique train labels
# unique, counts = np.unique(y_train, return_counts=True)
# print("Train labels: ", dict(zip(unique, counts)))

# count the number of unique test labels
# unique, counts = np.unique(y_test, return_counts=True)
# print("\nTest labels: ", dict(zip(unique, counts)))

#num_labels = len(np.unique(y_train))

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
# print(input_size)
image_size = x_train.shape[1]
input_size = image_size * image_size

x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
print("--- %s seconds ---" % (time.time() - start_time))

# x_test = np.reshape(x_test, [-1, input_size])
# x_test = x_test.astype('float32') / 255

#np.savetxt('x_test_large.txt', x_train, delimiter=',')




#print(x_train[0])
# model = Sequential()
# model.add(Dense(hidden_units, input_dim=input_size))
# model.add(Activation('relu'))
# model.add(Dense(num_labels))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', 
#               optimizer='adam',
#               metrics=['accuracy'])

# model.summary()

# model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

# loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
# print("\nTest accuracy: %.1f%%" % (100.0 * acc))

# model.save("m1024")



# print(x_test[0])

#weights = model.get_weights()
#weights_file = 'w1024.txt'


# with open(weights_file, 'w') as file:
#     for weight_array in weights:
#         np.savetxt(file, weight_array, delimiter=',')
#         file.write('\n')  # Add a newline after each weight array


# biases = [weights[i + 1] for i in range(0, len(weights), 2)]


# biases_file = 'b1024.txt'


# with open(biases_file, 'w') as file:
#     for i, bias in enumerate(biases):
#         bias_str = ', '.join(str(b) for b in bias)
#         file.write(f'Biases for layer {i+1}: {bias_str}\n')

r = model.predict(x_train)
print("--- %s seconds ---" % (time.time() - start_time))
# print(r)
# print(np.argmax(r[0]))

