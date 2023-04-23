import os

import cv2
import h5py

from keras.saving.legacy.model_config import model_from_json
from keras.models import load_model
from keras.utils import to_categorical
from path import Path
import numpy as np
from skimage.filters.thresholding import threshold_local

from src.segmentation import Segmentation
from src.dataloader_iam import DataLoaderIAM, Batch, Sample

from src.model import ModelIAM
from src.predict import Predict

from src.train import Train

print("baciu")





def train_main():
    loader = DataLoaderIAM(Path('D:/school-projects/year3sem1/licenta/Iam-dataset'), [0.90,0.95])
    char_list=loader.char_list
    "save the char list"
    f = open("../models/characters.txt", "w")
    f.write(''.join(char_list))
    f.close()
    img_w = 128
    img_h = 64
    max_text_len = 20
    modelIAM = ModelIAM(img_h, img_w, max_text_len,char_list=char_list)

    model = modelIAM.model
    model_predict = modelIAM.model_predict
    ckp, early_stopping = modelIAM.metrics()
    model_predict.summary()


    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    batch_size = 64########################era 64

    training_data = loader.train_samples[:]
    validation_data = loader.validation_samples[:]
    test_data=loader.test_samples[:]
    train=Train(img_h,img_w,max_text_len,model,model_predict,char_list,training_data,validation_data,test_data,ckp,early_stopping,batch_size)
    train.train()
    ####modelul se schimba in metoda??????????????????????????????//
    train.model.save('modelBinFinal.h5')
    with open('database/models/line_model_predictBinFinal.json', 'w') as f:
        f.write(train.model_predict.to_json())
    return train.model,train.model_predict

def main():
    """Main function."""
    #models,model_predict=train_main()
    #assert False

    segmentation=Segmentation()

    #train_main()


#add padding jos poate??????????????
    segmentation.predict_photo_text('D:/school-projects/year3sem1/licenta/summer/src/predictions/testing/new_Text.jpeg')

    #test_dataset()

    ##########
    #thinned = cv2.ximgproc.thinning(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))




def test_dataset():
    #read models

    print("start")
    with open('database/models/line_model_predictBinFinal.json', 'r') as f:
        l_model_predict = model_from_json(f.read())
    l_model_predict.load_weights('D:/school-projects/year3sem1/licenta/summer/src/database/epochs/Baciu BinFinal--11--1.833.h5')
    # read char list
    f = open("D:/school-projects/year3sem1/licenta/summer/src/database/characters.txt", "r")
    string = f.read()
    char_list = []
    char_list[:0] = string
    f=open('D:/school-projects/year3sem1/licenta/summer/src/database/datasplitTest.txt', 'r')
    test_data=[]
    for word_img_path in f:
        word,img_path=word_img_path.split(' word-split-path ')
        img_path=img_path[:-1]
        test_data.append(Sample(word,img_path))

    predict = Predict(64,128,l_model_predict, char_list,test_data=test_data)
    predict.testing()



if __name__ == '__main__':
    main()


