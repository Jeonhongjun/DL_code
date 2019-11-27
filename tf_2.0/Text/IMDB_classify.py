%pylab inline
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)
tf.executing_eagerly()

imdb = keras.datasets.imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

print('train sample : {}, Lable : {}'.format(len(x_train), len(x_test)))

# 단어와 정수 인덱스 매핑
word_index = imdb.get_word_index()
# 처음 몇 개 인덱스는 사전 정의 x
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(x_train[0])

# 영화 리뷰의 길이를 모두 맞추는 작업 : pad_sequence

x_train = keras.preprocessing.sequence.pad_sequences(x_train, value = word_index['<PAD>'], padding = 'post', maxlen=256)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, value = word_index['<PAD>'], padding = 'post', maxlen=256)

# 입력크기는 영화 리뷰 데이터셋의 어휘 크기
vocab_size = 10000

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 16, input_shape = (None,)),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
    ])

model.summary()

model.compile(optimizer = tf.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])

x_val = x_train[:10000]
x_train = x_train[10000:]

y_val = y_train[:10000]
y_train = y_train[10000:]

callbacks = keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5, min_delta = 0.00001)

history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 20, batch_size = 256, callbacks = [callbacks])

result = model.evaluate(x_test, y_test)

print(result)

history_dict = history.history

history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.xlabel('Accuracy')
plt.legend()

plt.show()
