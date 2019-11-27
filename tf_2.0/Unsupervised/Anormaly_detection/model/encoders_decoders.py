from tensorflow import keras
import numpy as np
from pytorch.Unsupervised.Anormaly_detection.utils import get_channels_axis

def conv_encoder(input_side = 3, n_channels = 3, representation_dim = 256, representation_activation = 'tanh', intermediate_activation = 'relu'):

    filter_size = 64
    input_shape = (n_channels, input_side, input_side) if get_channels_axis() == 1 else (input_side, input_side, n_channels)

    x_in = Input(shape=input_shape)
    encoder = x_in

    #downsample x0.5

    encoder = keras.layers.Conv2D(fiter_size, kernel_size = (3, 3), strides=(2,2), padding = 'same')(encoder)
    encoder = keras.layers.BatchNormalization(axis = get_channels_axis())(encoder)
    encoder = keras.layers.Activation(intermediate_activation)(encoder)

    #downsample x0.5

    encoder = keras.layers.Conv2D(fiter_size*2, kernel_size = (3, 3), strides=(2,2), padding = 'same')(encoder)
    encoder = keras.layers.BatchNormalization(axis = get_channels_axis())(encoder)
    encoder = keras.layers.Activation(intermediate_activation)(encoder)

    #downsample x0.5

    encoder = keras.layers.Conv2D(fiter_size*4, kernel_size = (3, 3), strides=(2,2), padding = 'same')(encoder)
    encoder = keras.layers.BatchNormalization(axis = get_channels_axis())(encoder)
    encoder = keras.layers.Activation(intermediate_activation)(encoder)

    if input_side == 64:
        encoder = keras.layers.Conv2D(fiter_size*8, kernel_size = (3, 3), strides=(2,2), padding = 'same')(encoder)
        encoder = keras.layers.BatchNormalization(axis = get_channels_axis())(encoder)
        encoder = keras.layers.Activation(intermediate_activation)(encoder)

    encoder = keras.layers.Flatten()(encoder)
    representation = keras.layers.Dense(representation_dim, activation = representation_activation)(encoder)

    return Model(x_in, representation)

def conv_decoder(output_side = 32, n_channels = 3, representation_dim = 256, activation = 'relu'):

    filter_size = 64

    representation_in = Input(shape = (representation_dim,))

    decoder = Dense(filter_size * 4 * 4 * 4)(representation_in)
    decoder = BatchNormalization(axis=-1)(decoder)
    decoder = keras.layers.Activation(activation)(decoder)

    conv_shape = (filter_size*4, 4, 4) if get_channels_axis() == 1 else (4, 4, filter_size * 4)
    decoder = keras.reshape(conv_shape)(decoder)

    # upsample x2.0
    decoder = keras.layers.Conv2DTr(filter_size*2, kernel_size = (3, 3), strides = (2,2), padding = 'same')(decoder)
    decoder = keras.layers.BatchNormalization(axis=get_channels_axis())(decoder)
    decoder = keras.layers.Activation(activation)

    # upsample x2.0
    decoder = keras.layers.Conv2DTr(filter_size, kernel_size = (3, 3), strides = (2,2), padding = 'same')(decoder)
    decoder = keras.layers.BatchNormalization(axis=get_channels_axis())(decoder)
    decoder = keras.layers.Activation(activation)(decoder)

    if output_size == 64:
        # upsample x2.0
        decoder = keras.layers.Conv2DTr(filter_size, kernel_size = (3, 3), strides = (2,2), padding = 'same')(decoder)
        decoder = keras.layers.BatchNormalization(axis=get_channels_axis())(decoder)
        decoder = keras.layers.Activation(activation)(decoder)

    decoder = keras.layers.Conv2DTr(n_channels, kernel_size = (3, 3), strides = (2,2), padding = 'same')(decoder)
    decoder = keras.layers.Activation('tanh')(decoder)

    return Model(representation_in, decoder)
