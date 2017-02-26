import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2

from keras import backend as K
K.set_image_dim_ordering('tf')

def nvidia(input_shape, dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, name='conv_1', subsample=(2, 2)))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Convolution2D(36, 5, 5, name='conv_2', subsample=(2, 2)))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Convolution2D(48, 5, 5, name='conv_3', subsample=(2, 2)))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, name='conv_4', subsample=(1, 1)))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, name='conv_5', subsample=(1, 1)))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Dense(1))

    return model


def nvidia_with_regularizer(input_shape, dropout):
    INIT = 'glorot_uniform'
    reg_val = 0.01

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT, W_regularizer=l2(reg_val)))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Dense(10))
    model.add(ELU())

    model.add(Dense(1))

    return model