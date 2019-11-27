

from tensorflow import keras

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor = 'val_acc',
        patience = 1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath = 'my_model.h5',
        monitor = 'val_loss',
        save_best_only = True,
    )
]

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metric = ['acc'])

model.fit(
        x, y,
        epochs = 10,
        batch_size = 64,
        callbacks = callbacks_list,
        validation_data = (x_val, y_val)
        )

callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_acc',
        patience = 1,
        factor = '0.1'
    )
]

model.fit(
        x, y,
        epochs = 10,
        batch_size = 64,
        callbacks = callbacks_list,
        validation_data = (x_val, y_val)
        )
