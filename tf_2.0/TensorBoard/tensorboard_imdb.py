import tensorflow as tf

keras = tf.keras

tf.executing_eagerly()

max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = max_features)

x_train = keras.preprocessing.sequence.pad_sequences(x_train,
    #padding = 'pre'
    #value = -1
    maxlen = max_len
    )

x_test = keras.preprocessing.sequence.pad_sequences(x_test,
    #padding = 'pre'
    #value = -1
    maxlen = max_len
    )

model = keras.models.Sequential([
        keras.layers.Embedding(max_features, 128, input_length = max_len, name = 'embed'),
        keras.layers.Conv1D(32, 7, activation = 'relu'),
        keras.layers.MaxPooling1D(5),
        keras.layers.Conv1D(32, 7, activation = 'relu'),
        keras.layers.GlobalMaxPooling1D(), #여러 개의 벡터 정보 중 가장 큰 벡터를 골라서 반환합니다.즉 문맥을 보면서 주요 특징을 뽑아내는 것
        keras.layers.Dense(1)
        ])

model.summary()

model.compile(optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['acc']
    )

callbacks = [
    keras.callbacks.TensorBoard(
    log_dir = 'my_log_dir',
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq='epoch')]

history = model.fit(x_train, y_train,
    epochs = 5,
    batch_size = 128,
    validation_split = 0.2,
    callbacks = callbacks)

from keras.utils import plot_model

plot_model(model, show_shapes = True, to_file = 'model.png')
