import tensorflow as tf
from tensorflow import keras
import datetime as dt

# tf.enable_eager_execution()
tf.executing_eagerly()

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.divide(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10))))
# tf.cast : type 지정
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
# random_flip_left_right : 이미지를 임의로 좌우로 뒤집음.

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000).shuffle(10000)
valid_dataset = valid_dataset.map(lambda x, y: (tf.divide(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10))))
valid_dataset = valid_dataset.repeat()
# repeat 은 데이터셋을 읽다가 마지막으로 도달했을 경우 다시 처음부터 조회하는 것.
train_dataset.take(1)


class CIFAR10_MODEL(keras.Model):

    def __init__(self):
        super(CIFAR10_MODEL, self).__init__(name='cifar_cnn')
        self.conv1 = keras.layers.Conv2D(
            64, 5, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.initializers.VarianceScaling,
            kernel_regularizer=keras.regularizers.l2(l=0.001))
        # kernel_initialize : 가중치 초기화 방법
        self.max_pool2d = keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), padding='same')
        self.max_norm = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(
            64, 5, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.initializers.VarianceScaling,
            kernel_regularizer=keras.regularizers.l2(l=0.001))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(
            750, activation=tf.nn.relu,
            kernel_initializer=tf.initializers.VarianceScaling,
            kernel_regularizer=keras.regularizers.l2(l=0.001))
        self.Dropout = keras.layers.Dropout(0.5)
        self.fc2 = keras.layers.Dense(10)
        self.softmax = keras.layers.Softmax()

    def call(self, x):
        x = self.max_pool2d(self.conv1(x))
        x = self.max_norm(x)
        x = self.max_pool2d(self.conv2(x))
        x = self.max_norm(x)
        x = self.flatten(x)
        x = self.Dropout(self.fc1(x))
        x = self.fc2(x)

        return self.softmax(x)


model = CIFAR10_MODEL()
model.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    , #keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001, patience=5)
    ]

hist = model.fit(train_dataset, epochs = 100, steps_per_epoch = 1500, validation_data = valid_dataset, validation_steps=5, callbacks = callbacks)

# 모델 학습 과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 사용하기
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))
