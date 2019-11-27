from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras  = tf.keras

import tensorflow_datasets as tfds

SPLIT_WEIGHTS = (8, 1, 1)

splilts = tfds.Split.TRAIN.subsplit(weighted=SPILT_WEIGHTS) # Train, Test, vaild set 나누는 code

(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', split = list(splilts),
    with_info = True, as_supervised=True)

print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()

# Relu 가 채널에 적용될 때에는 필연적으로 채널의 정보를 잃어버리게 됨.

IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

batch_size = 64
shuffle_buffer_size = 1000

train_batches = train.shuffle(shuffle_buffer_size).batch(batch_size)
validation_batches = validation.batch(batch_size)
test_batches = test.batch(batch_size)

for image_batch, label_batch in train_batches.take(1):
    pass

image_batch.shape

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# MobileNet V2 를 이용한 Pre-trained model

base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top=False, weights = 'imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape) # feature extractor

base_model.trainable = False

base_model.summary()

model  = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1),
    ])

base_learning_rate = 0.0001

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss = 'binary_crossentropy', metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

num_train, num_val, num_test = (metadata.splits['train'].num_examples*weight/10 for weight in SPLIT_WEIGHTS)

initial_epochs = 10
steps_per_epoch = round(num_train)//batch_size
validation_steps = 20

initial_loss, initial_accuracy = model.evaluate(validation_batches, steps = validation_steps)

print('initial loss : {:.2f}'.format(initial_loss))
print('initial accuracy : {:.2f}'.format(initial_accuracy))

callbacks = [keras.callbacks.ReduceLROnPlateau(patience = 5, factor = 0.5, monitor = 'val_loss')]

history = model.fit(train_batches,
    epochs = initial_epochs,
    validation_data = validation_batches,
    # callbacks = callbacks
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# loss & acc Graph

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Fine tuning

base_model.trainable = True

# MobileNet 의 layer 수
print("Number of layers in the base model: ", len(base_model.layers))

# 학습하지않는 Fine tune layer 수
none_fine_tune = 100

# none_fine_tune 이전 Freeze
for layer in base_model.layers[:none_fine_tune]:
  layer.trainable =  False

model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch = initial_epochs,
                         validation_data=validation_batches)

# loss & acc Graph

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
