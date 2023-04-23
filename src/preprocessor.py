import pickle
import random
from path import Path
from typing import Tuple

import cv2
import lmdb
import numpy as np
from src.dataloader_iam import Batch




class Preprocessor:
    def __init__(self,img_h=None,img_w=None,type='predict') :
        # when padding is on, we need dynamic width enabled
        self.img_h=img_h
        self.img_w=img_w
        self.env=lmdb.open(str('D:/school-projects/year3sem1/licenTa/Iam-dataset/lmdb'), readonly=True)
        self.type=type



    def get_img(self,file_path) -> np.ndarray:
        """
        read an image and convert it to 3-channel (BGR)
        """
        with self.env.begin() as image_opener:
            bn = Path(file_path).basename()
            data_img = image_opener.get(bn.encode("ascii"))
            img = pickle.loads(data_img)
            img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)# make the photo in 3 dimensions, not 2--if error
        return img


    def preprocess_data(self,path):
        """ Pre-process image"""
        if self.type == 'predict':
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.fix_size(img, self.img_w, self.img_h)

            img = np.clip(img, 0, 255)
            img = np.uint8(img)


            #cv2.imshow("dada",img)
            #cv2.waitKey()

            # normalize the image
            img = img.astype(np.float32)
            img /= 255
        else:
            img=self.get_img(path)
            #/////////////////
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)#poate la predict ar trebui sa fie inainte de ingrosare
            im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img=im_bw#ingrosare

            img = self.fix_size(img, self.img_w, self.img_h)


            img = np.clip(img, 0, 255)
            img = np.uint8(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #thinned = cv2.ximgproc.thinning(img)
            #cv2.imshow("dada",thinned)
            #cv2.waitKey()

            #normalize the image
            img = img.astype(np.float32)
            img /= 255



        return img



    def add_padding(self,img, old_width, old_height, new_w, new_h):#sa puna in ambele parti acceasi distanta
        """
        add a uniform amount of padding to an image
        """
        w1, w2 = int((new_w - old_width) / 2), int((new_w - old_width) / 2) + old_width
        h1, h2 = int((new_h - old_height) / 2), int((new_h - old_height) / 2) + old_height
        img_pad = np.ones([new_h, new_w]) * 255 #padding black#############
        img_pad[h1:h2, w1:w2] = img
        return img_pad

    def fix_size(self,img, target_width, target_height):
        """
        we are looking of the image sizes and the target sizes
        we will add padding to resolve the issues

        """
        height, width = img.shape[0],img.shape[1]
        if width < target_width and height < target_height:
            img = self.add_padding(img, width, height, target_width, target_height)
        elif width < target_width and height >= target_height:
            new_h = target_height
            new_w = int(width * new_h / height)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_width, target_height)
        elif width >= target_width and height < target_height:
            new_w = target_width
            new_h = int(height * new_w / width)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_width, target_height)


        else:
            '''width>=target_width and height>=target_height '''
            ratio = max(width / target_width, height / target_height)
            new_w = max(min(target_width, int(width / ratio)), 1)
            new_h = max(min(target_height, int(height / ratio)), 1)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_width, target_height)
        return img
