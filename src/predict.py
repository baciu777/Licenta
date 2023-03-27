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
        #pred_texts = self.decode_label_keras(net_out_value)
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
        return outstr

    def decode_label_keras(self, out):
        #out1 = K.get_value(
        #    K.ctc_decode(out[:, -30:, :], input_length=np.ones(out.shape[0]) * 30,
        #                 greedy=False,beam_width=1000)[0][0])
        #print(out1.shape)

        out2 = K.get_value(
            K.ctc_decode(out[:, -30:, :], input_length=np.ones(out.shape[0]) * 30,
                         greedy=True)[0][0])
        
        # see the results

        outstr = ''
        #for p in out1[0]:
        #    if int(p) != -1:
        #        outstr += self.char_list[p]
        #print("false "+ outstr)

        outstr = ''
        for p in out2[0]:
            if int(p) != -1:
                outstr += self.char_list[p]
        #spell=SpellChecker()

        #print("true "+spell.correction(outstr))
        return outstr


    def dilate_from_black(self,image,size_1,size_2):
        kernel = np.ones((size_1,size_2), np.uint8)
        dilated1 = cv2.dilate(image, kernel, iterations=1)
        return dilated1

    def spot_lines(self, dilated1, img):
        img=img.copy()
        (contours, heirarchy) = cv2.findContours(dilated1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # (x, y, w, h)

        for ctr in sorted_contours_lines:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(img, (x, y), (x + w, y + h), (40, 100, 250), 2)
        return img,sorted_contours_lines

    def word_image(self,x,y,word,img3):
        x2, y2, w2, h2 = cv2.boundingRect(word)
        image=[x + x2, y + y2, x + x2 + w2, y + y2 + h2]
        cv2.rectangle(img3, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 255, 100), 2)
        return image
    def sorted_contour_words(self,line,dilated2):
        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y + h, x:x + w]

        # draw contours on each word
        (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])
        return x,y,sorted_contour_words

    def crop(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=6)
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        ROI=image#--------------------
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ROI = image[y:y + h, x:x + w]
            cv2.imwrite('../src/predictions/a.png', ROI)
            break
        return ROI

    def black_and_white(self,image):
        # black and white

        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)###128 sau 0???
        m=im_bw


        # Save the black and white image
        #cv2.imwrite("../src/predictions/aba.jpg", im_bw)

        return m#----------------------------------------------------------------

    def white_pen_black_background(self,image):
        # black and white

        white_black_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T = threshold_local(white_black_img, 11, offset=10, method="gaussian")
        warped = (white_black_img > T).astype("uint8") * 255
        #kernel = np.ones((1, 1), np.uint8)
        #opening = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel)
        #kernel = np.ones((1, 1), np.uint8)
        #white_black_img = cv2.dilate(warped, kernel, iterations=1)

        (thresh, im_bw) = cv2.threshold(warped, 80, 255, cv2.THRESH_BINARY_INV)#//////////////////80

        return im_bw

    def increase_lines_width(self,imgContrast):

        # increase line width
        kernel = np.ones((3,3), np.uint8)# era 1---2

        imgMorph = cv2.erode(imgContrast, kernel, iterations=1)

        #imgMorph=imgContrast







        return imgMorph



    def testing(self):
        """
        here we test our model
        """
        words_error = 0
        words_nr = 0
        chars_error = 0
        chars_nr = 0
        for content, image in self.test_data:#/////////////
            prediction = self.img_predict(image)
            print(prediction)
            print(content)
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
