import cv2
from keras.saving.legacy.model_config import model_from_json
from spellchecker import SpellChecker
from textblob import TextBlob

from src.predict import Predict


import numpy as np


class Prediction(object):

    def __init__(self):
        print("start")
        # read char list
        f = open("../model/list.txt", "r")
        string = f.read()
        char_list = []
        char_list[:0] = string
        self.char_list=char_list
        with open('../src/doc/line_model_predict.json', 'r') as f:
            self.l_model_predict = model_from_json(f.read())
        self.l_model_predict.load_weights('D:/school-projects/year3sem1/licenta/summer/src/doc/Baciu Hand--15--1.514.h5')


        self.predictObject = Predict(64,128,self.l_model_predict, self.char_list)



    def predict_photo_text(self,img):
        print("alo baza baza")
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #??????????????????
        img2=img
        print("ajung1")
        #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)        #??????????????????
        #AM SCOS DENOISINGUL NU FACEA BINE PE CAPTURE IMAGES
        #img2=cv2.convertScaleAbs(img2,beta=60)#--
        #img2=cv2.convertScaleAbs(img2,alpha=1.6)#--

        print("ajung2")



        white_pen_img=self.predictObject.white_pen_black_background(img2)

        cv2.imwrite("../src/predictions/process/white_pen_img.jpg", white_pen_img)
        dilated1=self.predictObject.dilate_from_black(white_pen_img,8,500)
        cv2.imwrite("../src/predictions/process/dilated1.jpg", dilated1)


        img2,sorted_contours_lines=self.predictObject.spot_lines(dilated1, img)
        cv2.imwrite("../src/predictions/process/lines.jpg", img2)


        dilated2=self.predictObject.dilate_from_black(white_pen_img,35,25)#words
        cv2.imwrite("../src/predictions/process/dilated2.jpg", dilated2)
        img3 = img.copy()
        text_predict=""
        for line in sorted_contours_lines:
            _,_,w,h=cv2.boundingRect(line)
            if h<50:
                continue
            x,y,sorted_contour_words=self.predictObject.sorted_contour_words(line,dilated2)

            for word in sorted_contour_words:
                _, _, w, h = cv2.boundingRect(word)
                if h < 45:
                    continue
                word=self.predictObject.word_image(x,y,word,img3)
                roi = img[word[1]:word[3], word[0]:word[2]]
                roi = self.predictObject.black_and_white(roi)
                roi = self.predictObject.increase_lines_width(roi)
                cv2.imwrite("../src/predictions/intermitent_word.jpg", roi)
                prediction = ""
                prediction = self.predictObject.img_predict("../src/predictions/intermitent_word.jpg")
                print(prediction)
                spell = SpellChecker()
                checked_prediction = spell.correction(prediction)
                if checked_prediction:
                    print("checked " + checked_prediction)
                text_predict+=prediction+" "
            text_predict+='\n'

        #sentence=TextBlob(text_predict)
        #text_predict=sentence.correct()
        print(text_predict)
        cv2.imwrite("../src/predictions/process/words.jpg", img3)
        return text_predict






    def predict_photo(self,filename):
        # read model



        image = cv2.imread(filename)
        image = self.predictObject.crop(image)
        m=image


        #cv2.imshow('matrix', m)
        #cv2.waitKey(0)
        cv2.imwrite('../src/predictions/yourNewImage.jpg', m)
        image=m
        image = self.predictObject.black_and_white(image)
        image = self.predictObject.increase_lines_width(image)
        # write
        cv2.imwrite('../src/predictions/out.png', image)

        prediction = self.predictObject.img_predict('D:\school-projects\year3sem1\licenta\summer\src\predictions\out.png')
        print(prediction)

        spell = SpellChecker()
        checked_prediction = spell.correction(prediction)
        print("true " + checked_prediction)
        return checked_prediction