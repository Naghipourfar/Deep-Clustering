from keras.layers import *
from keras.models import *
from keras.models import Model
from keras.optimizers import *

"""
    Created by Mohsen Naghipourfar on 9/1/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def create_model(width, height, channels):
    input_layer = Input(shape=(width, height, channels))
    conv_1 = Conv2D(filters=8, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same')(input_layer)
    conv_2 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool_1)
    conv_4 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool_2)
    conv_6 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv_5)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_6)

    conv_7 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool_3)
    conv_8 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv_7)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_8)

    latent = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool_4)

    upsample_1 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(latent))
    merge_1 = Concatenate(axis=3)([upsample_1, conv_8])
    conv_9 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(merge_1)
    conv_10 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv_9)

    upsample_2 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(conv_10))
    merge_2 = Concatenate(axis=3)([upsample_2, conv_6])
    conv_11 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(merge_2)
    conv_12 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv_11)

    upsample_3 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(conv_12))
    merge_3 = Concatenate(axis=3)([upsample_3, conv_4])
    conv_13 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(merge_3)
    conv_14 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv_13)

    upsample_4 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(conv_14))
    merge_4 = Concatenate(axis=3)([upsample_4, conv_2])
    conv_15 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(merge_4)
    conv_16 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv_15)

    conv_17 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same')(conv_16)

    model = Model(inputs=input_layer, outputs=conv_17)
    model.compile(optimizer=Adam(lr=1e-4), loss=hamming_dist)

    model.summary()

    return model


def hamming_dist(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))


if __name__ == '__main__':
    model = create_model(128, 128, 1)
