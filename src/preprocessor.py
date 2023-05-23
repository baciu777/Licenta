import pickle
from path import Path
import cv2
import lmdb
import numpy as np

from src.utils import dataset_path


class Preprocessor:
    def __init__(self,img_h=None,img_w=None,type='predict') :
        self.img_h=img_h
        self.img_w=img_w
        self.env=lmdb.open(str(dataset_path+'/lmdb'), readonly=True)
        self.type=type



    def get_img(self,file_path) -> np.ndarray:
        with self.env.begin() as image_opener:
            bn = Path(file_path).basename()
            data_img = image_opener.get(bn.encode("ascii"))
            img = pickle.loads(data_img)
            img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


    def preprocess_data(self,input_data):
        if self.type == 'predict':

            img = self.fix_size(input_data, self.img_w, self.img_h)

            img = np.clip(img, 0, 255)
            img = np.uint8(img)


            img = img.astype(np.float32)
            img /= 255
        else:
            img=self.get_img(input_data)
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img=im_bw
            img = self.fix_size(img, self.img_w, self.img_h)
            img = np.clip(img, 0, 255)
            img = np.uint8(img)

            img = img.astype(np.float32)
            img /= 255

        return img



    def add_padding(self,img, old_width, old_height, new_w, new_h):

        w1, w2 = int((new_w - old_width) / 2), int((new_w - old_width) / 2) + old_width
        h1, h2 = int((new_h - old_height) / 2), int((new_h - old_height) / 2) + old_height
        img_pad = np.ones([new_h, new_w]) * 255
        img_pad[h1:h2, w1:w2] = img
        return img_pad

    def fix_size(self,img, target_width, target_height):
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
            ratio = max(width / target_width, height / target_height)
            new_w = max(min(target_width, int(width / ratio)), 1)
            new_h = max(min(target_height, int(height / ratio)), 1)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_width, target_height)
        return img
