import itertools

import editdistance
import skimage.morphology
import cv2
from keras import backend as K
import numpy as np
from skimage.filters.thresholding import threshold_local
from spellchecker import SpellChecker

from src.preprocessor import Preprocessor
import tensorflow as tf
from queue import PriorityQueue

class Predict:
    def __init__(self,img_h,img_w,model_predict,char_list,test_data=None):
        self.img_h=img_h
        self.img_w=img_w
        if test_data is not None:
            self.preprocessor=Preprocessor(img_h, img_w,'test')
        else:
            self.preprocessor=Preprocessor(img_h,img_w,'predict')
        self.model_predict=model_predict
        self.char_list=char_list
        self.test_data=test_data

    def img_predict(self,file_path):

        img = self.preprocessor.preprocess_data(file_path)
        img = img.T

        if K.image_data_format() == 'channels_first':
            img = np.expand_dims(img, 0)
        else:
            img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        net_out_value = self.model_predict.predict(img)

        # use CTC decoder

        pred_texts = self.decode_label(net_out_value)


        #red_texts = self.decode_label_keras(net_out_value)#mult mai lent cu kerasescu
        #e fara corectie vezi


        return pred_texts


    def decode_label(self,out):
        out_best = list(np.argmax(out[0, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        #print(out_best)
        for c in out_best:
            if c < len(self.char_list):
                outstr += self.char_list[c]
        spell=SpellChecker()
        outstr1=spell.correction(outstr)
        if outstr1 is None:
            outstr1=outstr

        return outstr1

    def has_consecutive_duplicate(self,string):
        for i in range(1, len(string)):
            if string[i] == string[i - 1]:
                return True
        return False

    def decode_label_keras(self, out):
        out1 = K.get_value(
            K.ctc_decode(out[:, -30:, :], input_length=np.ones(out.shape[0]) * 30,
                         greedy=True,)[0][0])
        #print(out1.shape)
        outstr = ''
        for p in out1[0]:
            if int(p) != -1:
                outstr += self.char_list[p]
        if self.has_consecutive_duplicate(outstr) is False:
            print("da")
            out2 = K.get_value(
            K.ctc_decode(out[:, -30:, :], input_length=np.ones(out.shape[0]) * 30,
                       greedy=False,beam_width=100,top_paths=1)[0][0])
            outstr = ''
            for p in out2[0]:
                if int(p) != -1:
                    outstr += self.char_list[p]
        # see the results



        #spell=SpellChecker()

        #print("true "+spell.correction(outstr))
        return outstr



    def testing(self):
        """
        here we test our models
        """
        words_error = 0
        words_nr = 0
        chars_error = 0
        chars_nr = 0
        for content, image in self.test_data:#/////////////
            prediction = self.img_predict(image)
            print(content)
            print(prediction)
            print("-------------")
            if content != prediction:
                words_error += 1
            words_nr += 1

            chars_error += editdistance.eval(content, prediction)
            chars_nr += len(content)
            print('ED chars: ', chars_error)
            print('ED words: ', words_nr)

        print('CER: ', chars_error / chars_nr)
        print('WER: ', words_error / words_nr)
