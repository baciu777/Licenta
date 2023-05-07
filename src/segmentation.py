import threading

import cv2
from PIL import Image
from keras.saving.legacy.model_config import model_from_json
from skimage.filters.thresholding import threshold_local
from spellchecker import SpellChecker
from spello.model import SpellCorrectionModel
from textblob import TextBlob

from src.predict import Predict


import numpy as np


class Segmentation(object):

    def __init__(self):
        print("start")
        # read char list
        f = open("D:\school-projects\year3sem1\licenta\summer\src\database/characters.txt", "r")
        string = f.read()
        char_list = []
        char_list[:0] = string
        self.char_list=char_list
        with open('D:\school-projects\year3sem1\licenta\summer\src\database/models/line_model_predict.json', 'r') as f:
            self.l_model_predict = model_from_json(f.read())
        #self.l_model_predict.load_weights('D:\school-projects\year3sem1\licenta\summer\src\database/epochs/Baciu BinFinal--11--1.833.h5')
        self.l_model_predict.load_weights('D:\school-projects\year3sem1\licenta\summer\src\database/epochs/Baciu Hand--15--1.514.h5')


        self.predictObject = Predict(64,128,self.l_model_predict, self.char_list)



    def predict_photo_text(self,imgPath):
        print("alo baza baza")
        print(imgPath)
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2=img
        #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)        #??????????????????
        #AM SCOS DENOISINGUL NU FACEA BINE PE CAPTURE IMAGES
        #img2=cv2.convertScaleAbs(img2,beta=60)#--
        #img2=cv2.convertScaleAbs(img2,alpha=1.6)#--

        img_w,img_h,_=img.shape
        white_pen_img=self.white_pen_black_background(img2)

        cv2.imwrite("D:\school-projects\year3sem1\licenta\summer\src\predictions/process/white_pen_img.jpg", white_pen_img)
        dilated1=self.dilate_from_black(white_pen_img,int(img_h/400-1),500)#era 8,500
        cv2.imwrite("D:\school-projects\year3sem1\licenta\summer\src\predictions/process/dilated1.jpg", dilated1)


        img2,sorted_contours_lines=self.spot_lines(dilated1, img)
        cv2.imwrite("D:\school-projects\year3sem1\licenta\summer\src\predictions/process/lines.jpg", img2)


        dilated2=self.dilate_from_black(white_pen_img,int(img_h/140),int(img_w/110))#words##aici cred ca poti face scalat, assumption ca da width de un rand
        cv2.imwrite("D:\school-projects\year3sem1\licenta\summer\src\predictions/process/dilated2.jpg", dilated2)
        img3 = img.copy()
        text_predict=""
        for line in sorted_contours_lines:
            _,_,w,h=cv2.boundingRect(line)
            if h<int(img_h/30):#era 50.. ce facem aici? am presupus ca nu sunt mai mult de 30 de randuri pe foaie deci dimns minima a lunii e img_h/30
                continue
            x,y,sorted_contour_words=self.sorted_contour_words(line,dilated2)

            for word in sorted_contour_words:
                _, _, w, h = cv2.boundingRect(word)
                if h < int(img_h/40) and w <int(img_w/20):
                   continue
                word=self.word_image(x,y,word,img3)
                roi = img[word[1]:word[3], word[0]:word[2]]
                roi = self.black_and_white(roi)
                roi = self.increase_lines_width(roi)
                cv2.imwrite("D:\school-projects\year3sem1\licenta\summer\src\predictions\predictionWord/intermitent_word.jpg", roi)
                prediction = self.predictObject.img_predict("D:\school-projects\year3sem1\licenta\summer\src\predictions\predictionWord/intermitent_word.jpg")
                print(prediction)
                spell = SpellChecker()
                checked_prediction = spell.correction(prediction)
                if checked_prediction:
                    print("checked " + checked_prediction)

                text_predict+=prediction+" "
            text_predict+='\n'

        cv2.imwrite("D:\school-projects\year3sem1\licenta\summer\src\predictions/process/words.jpg", img3)
        return text_predict




    def white_pen_black_background(self,image):
        # black and white

        white_black_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T = threshold_local(white_black_img, 11, offset=10, method="gaussian")
        warped = (white_black_img > T).astype("uint8") * 255

        (thresh, im_bw) = cv2.threshold(warped, 80, 255, cv2.THRESH_BINARY_INV)#//////////////////80

        return im_bw


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

    def sorted_contour_words(self,line,dilated2):
        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y + h, x:x + w]

        # draw contours on each word
        (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])
        return x,y,sorted_contour_words
    def word_image(self,x,y,word,img3):
        x2, y2, w2, h2 = cv2.boundingRect(word)
        image=[x + x2, y + y2, x + x2 + w2, y + y2 + h2]
        cv2.rectangle(img3, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 255, 100), 2)
        return image


    def increase_lines_width(self,imgContrast):
        img_w_photo,img_h_photo=imgContrast.shape
        # increase line width
        kernel = np.ones((int(img_h_photo/40),int(img_w_photo/40)), np.uint8)

        imgMorph = cv2.erode(imgContrast, kernel, iterations=1)

        return imgMorph
    def black_and_white(self,image):
        # black and white

        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)###128 sau 0???
        m=im_bw

        return m#----------------------------------------------------------------