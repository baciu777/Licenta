import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Reshape, Lambda, BatchNormalization
from keras.layers import GRU
from keras.layers import add, concatenate
from keras import backend as K



tf.compat.v1.disable_eager_execution()


class ModelIAM:
    """Minimalistic TF models for HTR."""

    def __init__(self,img_h,img_w,max_text_len,char_list) -> None:
        self.img_h=img_h
        self.img_w=img_w
        self.max_text_len = max_text_len
        self.char_list=char_list
        if K.image_data_format() == 'channels_first':#sau channels_last
            input_shape = (1, img_w, img_h)
        else:
            input_shape = (img_w, img_h, 1)

        self.setup_model(input_shape)


    def setup_model(self,input_shape):
        # Make Network
        input_data = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)





        # Convolution layer (VGG)
        custom = Conv2D(64, (5, 5), padding='same',  kernel_initializer='he_normal')(input_data)  # (None, 128, 64, 64)
        custom = BatchNormalization()(custom)
        custom = Activation('relu')(custom)
        custom = MaxPooling2D(pool_size=(2, 2))(custom)  # (None,64, 32, 64)


        custom = Conv2D(128, (5, 5), padding='same', kernel_initializer='he_normal')(custom)  # (None, 64, 32, 128)
        custom = BatchNormalization()(custom)
        custom = Activation('relu')(custom)
        custom = MaxPooling2D(pool_size=(2, 2))(custom)  # (None, 32, 16, 128)

        custom = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(custom)  # (None, 32, 16, 256)
        custom = BatchNormalization()(custom)
        custom = Activation('relu')(custom)

        custom = Conv2D(256, (3, 3), padding='same',  kernel_initializer='he_normal')(custom)  # (None, 32, 16, 256)
        custom = BatchNormalization()(custom)
        custom = Activation('relu')(custom)
        custom = MaxPooling2D(pool_size=(1, 2))(custom)  # (None, 32, 8, 256)

        custom = Conv2D(512, (3, 3), padding='same',  kernel_initializer='he_normal')(custom)  # (None, 32, 8, 512)
        custom = BatchNormalization()(custom)
        custom = Activation('relu')(custom)

        custom = Conv2D(512, (3, 3), padding='same')(custom)  # (None, 32, 8, 512)
        custom = BatchNormalization()(custom)
        custom = Activation('relu')(custom)
        custom = MaxPooling2D(pool_size=(1, 2))(custom)  # (None, 32, 4, 512)

        custom = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal')(custom)  # (None, 32, 4, 512)
        custom = BatchNormalization()(custom)
        custom = Activation('relu')(custom)

        # CNN to RNN
        custom = Reshape(target_shape=((32, 2048)))(custom)  # (None, 32, 2048)
        custom = Dense(64, activation='relu', kernel_initializer='he_normal')(custom)  # (None, 32, 64)

        # RNN layer
        gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal')(custom)  # (None, 32, 512)
        gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(
            custom)
        reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_1b)

        gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
        gru1_merged = BatchNormalization()(gru1_merged)

        gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal')(gru1_merged)
        gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(
            gru1_merged)
        reversed_gru_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_2b)

        gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
        gru2_merged = BatchNormalization()(gru2_merged)

        num_classes = len(self.char_list) + 1

        # transforms RNN output to character activations:
        custom = Dense(num_classes, kernel_initializer='he_normal')(gru2_merged)  # (None, 32, 80)
        y_pred = Activation('softmax', name='softmax')(custom)

        labels = Input(name='the_labels', shape=[self.max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        # loss function
        loss_out = Lambda(ctc_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length]
        )
        self.model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        self.model_predict = Model(inputs=input_data, outputs=y_pred)




    def metrics(self):
        ckp = ModelCheckpoint(
            filepath='database/epochs/' + 'Baciu BinFinal' + '--{epoch:02d}--{val_loss:.3f}.h5', monitor='val_loss',
            verbose=1, save_best_only=True, save_weights_only=True
        )
        earlystop = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='min'
        )
        return ckp,earlystop

def ctc_func(args):
    """
    const function using connectionist temporal classification
    y_pred is the prediction
    labels is the true result
    """
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)