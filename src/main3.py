import os

import cv2
import h5py

from keras.saving.legacy.model_config import model_from_json
from keras.models import load_model
from keras.utils import to_categorical
from path import Path
import numpy as np
from skimage.filters.thresholding import threshold_local

from src.prediction import Prediction
from src.dataloader_iam import DataLoaderIAM, Batch, Sample

from src.model import ModelIAM
from src.predict import Predict

from src.train import Train

print("baciu")





def train_main():
    loader = DataLoaderIAM(Path('D:/school-projects/year3sem1/licenta/Iam-dataset'), [0.90,0.95])
    char_list=loader.char_list
    "save the char list"
    f = open("../model/List.txt", "w")
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

    batch_size = 32########################era 64

    training_data = loader.train_samples[:]
    validation_data = loader.validation_samples[:]
    test_data=loader.test_samples[:]
    train=Train(img_h,img_w,max_text_len,model,model_predict,char_list,training_data,validation_data,test_data,ckp,early_stopping,batch_size)
    model=train.testing()#####verifica daca e nevoie de return
    model.save('modelNEW.h5')
    with open('doc/line_model_predictNEW.json', 'w') as f:
        f.write(model_predict.to_json())
    return model,model_predict

def main():
    """Main function."""
    #model,model_predict=train_main()
    #assert False

    #object=Prediction()

    train_main()


#add padding jos poate??????????????
    #object.predict_photo_text('D:/school-projects/year3sem1/licenta/summer/src/predictions/want.jpg')
    #test_dataset()





def test_dataset():
    #read model

    print("start")
    with open('../src/doc/line_model_predict1.json', 'r') as f:
        l_model_predict = model_from_json(f.read())
    l_model_predict.load_weights('D:/school-projects/year3sem1/licenta/summer/src/doc/Baciu Hand--15--1.514.h5')
    # read char list
    f = open("../model/list.txt", "r")
    string = f.read()
    char_list = []
    char_list[:0] = string
    f=open('../model/testDataSplit.txt', 'r')
    test_data=[]
    for word_img_path in f:
        word,img_path=word_img_path.split(' word-split-path ')
        img_path=img_path[:-1]
        test_data.append(Sample(word,img_path))

    predict = Predict(64,128,l_model_predict, char_list,test_data=test_data)
    predict.testing()



if __name__ == '__main__':
    main()


