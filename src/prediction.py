import itertools
import editdistance
from keras import backend as K
import numpy as np
from spellchecker import SpellChecker

from src.preprocessor import Preprocessor

class Prediction:
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

    def img_predict(self,input_data):
        img = self.preprocessor.preprocess_data(input_data)
        img = img.T

        if K.image_data_format() == 'channels_first':
            img = np.expand_dims(img, 0)
        else:
            img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        net_out_value = self.model_predict.predict(img)

        pred_texts = self.decode_label(net_out_value)

        return pred_texts


    def decode_label(self,out):
        out_best = list(np.argmax(out[0, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(self.char_list):
                outstr += self.char_list[c]
        spell=SpellChecker()
        outstr1=spell.correction(outstr)
        if outstr1 is None:
            outstr1=outstr
            if outstr[0].isupper():
                outstr1[0]=outstr1[0].upper()
        print(outstr+"   "+outstr1)
        return outstr1



    def testing(self):
        """
        here we test our models
        """
        words_error = 0
        words_nr = 0
        chars_error = 0
        chars_nr = 0
        for content, image_path in self.test_data:

            prediction = self.img_predict(image_path)
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


