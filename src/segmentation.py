import cv2
from keras.saving.legacy.model_config import model_from_json
from spellchecker import SpellChecker
import numpy as np
from src.prediction import Prediction
from src.utils import database_path


class Segmentation(object):

    def __init__(self):
        print("start")
        # read char list
        f = open(database_path+"/characters.txt", "r")
        string = f.read()
        char_list = []
        char_list[:0] = string
        self.char_list=char_list
        with open(database_path+'/models/line_model_predict.json', 'r') as f:
            self.l_model_predict = model_from_json(f.read())
        #self.l_model_predict.load_weights('D:\school-projects\year3sem1\licenta\summer\src\database/epochs/Baciu BinFinal--11--1.833.h5')
        self.l_model_predict.load_weights(database_path+'/epochs/Baciu Hand--15--1.514.h5')
        self.l_model_predict._make_predict_function()
        self.predict_object = Prediction(64, 128, self.l_model_predict, self.char_list)



    def predict_photo_text(self, img):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_copy=img.copy()
        img_w,img_h,_=img.shape
        white_pen_img=self.white_pen_black_background_scan(img_copy)

        dilated1=self.dilate_from_black(white_pen_img,int(img_h/400-1),500)

        sorted_contours_lines=self.spot_lines(dilated1)

        dilated2=self.dilate_from_black(white_pen_img,int(img_h/115),int(img_w/180))#words
        img3 = img.copy()
        text_predict=""
        for line in sorted_contours_lines:
            _,_,w,h=cv2.boundingRect(line)
            if h<int(img_h/40):
                continue
            x,y,sorted_contour_words=self.sorted_contour_words(line,dilated2)

            for word in sorted_contour_words:
                _, _, w, h = cv2.boundingRect(word)
                word=self.word_image(x,y,word,img3)
                roi = img[word[1]:word[3], word[0]:word[2]]
                roi = self.black_and_white(roi)
                roi = self.increase_lines_width(roi)
                prediction = self.predict_object.img_predict(roi)
                punctuations=".:;?!,"
                if prediction in punctuations:
                    text_predict+=prediction
                else:
                    text_predict+=prediction+" "

            text_predict+='\n'

        return text_predict

    def white_pen_black_background_scan(self,img2):
        gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
        mask = cv2.adaptiveThreshold(blurred,
                                     255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     31,
                                     10)
        inverted_mask = cv2.bitwise_not(mask)
        return inverted_mask

    def dilate_from_black(self,image,size_1,size_2):
        kernel = np.ones((size_1,size_2), np.uint8)
        dilated1 = cv2.dilate(image, kernel, iterations=1)
        return dilated1

    def spot_lines(self, dilated1):
        (contours, heirarchy) = cv2.findContours(dilated1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
        return sorted_contours_lines

    def sorted_contour_words(self,line,dilated2):
        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y + h, x:x + w]

        # draw contours on each word
        (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Filter out nested contours
        filtered_contours = []
        for cntr in cnt:
            (x_cntr, y_cntr, w_cntr, h_cntr) = cv2.boundingRect(cntr)
            is_nested = False
            for other_cntr in cnt:
                if cntr is not other_cntr:
                    (x_other, y_other, w_other, h_other) = cv2.boundingRect(other_cntr)
                    if x_cntr >= x_other and y_cntr >= y_other and x_cntr + w_cntr <= x_other + w_other and y_cntr + h_cntr <= y_other + h_other:
                        is_nested = True
                        break
            if not is_nested:
                filtered_contours.append(cntr)
        sorted_contour_words = sorted(filtered_contours, key=lambda cntr: cv2.boundingRect(cntr)[0])

        return x,y,sorted_contour_words
    def word_image(self,x,y,word,img3):
        x2, y2, w2, h2 = cv2.boundingRect(word)
        image=[x + x2, y + y2, x + x2 + w2, y + y2 + h2]
        cv2.rectangle(img3, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 255, 100), 2)
        return image


    def increase_lines_width(self, img_contrast):
        img_w_photo,img_h_photo=img_contrast.shape
        kernel = np.ones((int(img_h_photo/40),int(img_w_photo/40)), np.uint8)
        img_morph = cv2.erode(img_contrast, kernel, iterations=1)
        return img_morph

    def black_and_white(self,image):
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)###128 sau 0???
        m=im_bw
        return m

