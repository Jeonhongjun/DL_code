from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 10))
for i in range(25) :
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Dense net
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape = (28, 28)),
#     keras.layers.Dense(128, activation = tf.nn.relu),
#     keras.layers.Dense(10, activation = tf.nn.softmax)
#     ])

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

model = keras.Sequential([
    keras.layers.Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = tf.nn.relu, input_shape= (28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation = tf.nn.softmax),
    ])
model.summary()

model.compile(optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'])

early_stoping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto')
# monitor : 관찰하고자 하는 항목입니다. ‘val_loss’나 ‘val_acc’가 주로 사용됩니다.
# min_delta : 개선되고 있다고 판단하기 위한 최소 변화량을 나타냅니다. 만약 변화량이 min_delta보다 적은 경우에는 개선이 없다고 판단합니다.
# patience : 개선이 없다고 바로 종료하지 않고 개선이 없는 에포크를 얼마나 기다려 줄 것인 가를 지정합니다. 만약 10이라고 지정하면 개선이 없는 에포크가 10번째 지속될 경우 학습일 종료합니다.
# verbose : 얼마나 자세하게 정보를 표시할 것인가를 지정합니다. (0, 1, 2)
# mode : 관찰 항목에 대해 개선이 없다고 판단하기 위한 기준을 지정합니다. 예를 들어 관찰 항목이 ‘val_loss’인 경우에는 감소되는 것이 멈출 때 종료되어야 하므로, ‘min’으로 설정됩니다.
# auto : 관찰하는 이름에 따라 자동으로 지정합니다.
# min : 관찰하고 있는 항목이 감소되는 것을 멈출 때 종료합니다.
# max : 관찰하고 있는 항목이 증가되는 것을 멈출 때 종료합니다.

a = model.fit(train_images, train_labels, epochs = 1, batch_size = 64, callbacks = [early_stoping])
np.array(a)
predicted = model.predict(test_images)

test_accuracy = model.evaluate(test_images, test_labels)
