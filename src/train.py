import itertools
import random

import editdistance
from keras.callbacks import History

from src.dataloader_iam import DataLoaderIAM
from src.preprocessor import Preprocessor
from path import Path
from keras import backend as K
import numpy as np


class Train:
    def __init__(self, img_h, img_w, max_text_len, model, model_predict, char_list, train_samples, validation_samples,
                 test_samples, ckp, early_stopping, batch_size):
        self.img_h = img_h
        self.img_w = img_w
        self.max_text_len = max_text_len
        self.model = model
        self.model_predict = model_predict
        self.char_list = list(char_list)
        self.training_data = train_samples
        self.validation_data = validation_samples
        self.test_data = test_samples
        self.ckp = ckp
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.preprocessor = Preprocessor(img_h, img_w, 'train')
        ###


    def build_data(self, samples_size, img_h, img_w, samples):
        imgs = np.zeros((samples_size, img_h, img_w))
        texts = []
        random.shuffle(samples)
        count = 0
        # process image, append word
        for i, (word, file_path) in enumerate(samples):
            img = self.preprocessor.preprocess_data(file_path)
            imgs[i, :, :] = img
            count += 1
            if count % 100 == 0:
                print(count)
            texts.append(word)
        return texts, imgs

    def train(self):

        size_train = len(self.training_data)

        size_vali = len(self.validation_data)

        training_generator = self.data_generator(self.training_data, self.img_h, self.img_w, self.max_text_len,
                                                 size_train,
                                                 batch_size=self.batch_size)
        validation_generator = self.data_generator(self.validation_data, self.img_h, self.img_w, self.max_text_len,
                                                   size_vali,
                                                   batch_size=self.batch_size)
        history = self.model.fit_generator(generator=training_generator,
                                           steps_per_epoch=size_train // self.batch_size,
                                           epochs=20,
                                           validation_data=validation_generator,
                                           validation_steps=size_vali // self.batch_size,
                                           callbacks=[self.ckp, self.early_stopping], verbose=1)

    def data_generator(self, samples, img_h, img_w, max_text_len, input_size, batch_size):
        """
        data generator that iterates through images and text until the batch_size is reached,
        where we send input data,the labels,data_length and labels_length
        """
        # if all images have the same sizes, it is appropriate to use the same input_length
        input_length = 30  ###############is the length of the input sequence in time steps

        print(batch_size)
        print(len(samples))
        texts, imgs = self.build_data(len(samples), img_h, img_w, samples)
        count_total = 0
        while True:
            # width and height are backwards from typical Keras convention - RNN
            X_train, Y_train, data_length, label_length = self.batch_data(batch_size, img_h, img_w, max_text_len,
                                                                          input_length)
            counter = 0
            # print('while true')
            for content, image in zip(texts, imgs):
                image = image.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(image, 0)
                else:
                    img = np.expand_dims(image, -1)
                X_train[counter] = img
                Y_train[counter, :len(content)] = self.text_to_labels(content)
                label_length[counter] = len(content)
                counter += 1

                if (counter % batch_size == 0):  # end of a batch
                    # print("batch")
                    counter = 0
                    inputs = {
                        'the_input': X_train,
                        'the_labels': Y_train,
                        'input_length': data_length,
                        'label_length': label_length,
                    }
                    outputs = {'ctc': np.zeros([batch_size])}
                    yield (inputs, outputs)
                    count_total += 1
                    X_train, Y_train, data_length, label_length = self.batch_data(batch_size, img_h, img_w,
                                                                                  max_text_len, input_length)

    def batch_data(self, batch_size, img_h, img_w, max_text_len, input_length):
        # x_data will hold the images data for a batch
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([batch_size, 1, img_w, img_h])
        else:
            X_data = np.ones([batch_size, img_w, img_h, 1])

        data_length = np.ones((batch_size, 1)) * input_length  # [30,30,30 ,....30]
        # vezi pe aici
        # y_data will hold the texts for a batch
        Y_data = np.zeros([batch_size, max_text_len])
        label_length = np.zeros((batch_size, 1))

        return X_data, Y_data, data_length, label_length

    def text_to_labels(self, text):
        """
        method is used to convert a given text into a list of labels,
        where each label represents the index of the corresponding character in the char_list
        """
        return list(map(lambda x: self.char_list.index(x), text))
