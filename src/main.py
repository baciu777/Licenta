from keras.saving.legacy.model_config import model_from_json
from path import Path
from src.segmentation import Segmentation
from src.dataloader_iam import DataLoaderIAM,  Sample
from src.model import ModelIAM
from src.prediction import Prediction
from src.train import Train
from src.utils import database_path

print("baciu")





def train_main():
    loader = DataLoaderIAM([0.90,0.95])
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

    batch_size = 64

    training_data = loader.train_samples
    validation_data = loader.validation_samples
    test_data=loader.test_samples
    train=Train(img_h,img_w,max_text_len,model,model_predict,char_list,training_data,validation_data,test_data,ckp,early_stopping,batch_size)
    train.train()
    train.model.save('modelBinFinal.h5')
    with open(database_path+'/models/line_model_predictBinFinal.json', 'w') as f:
        f.write(train.model_predict.to_json())
    return train.model,train.model_predict

def main():
    """Main function."""
    #models,model_predict=train_main()

    segmentation=Segmentation()

    #train_main()

    segmentation.predict_photo_text('D:/school-projects/year3sem1/licenta/summer/src/predictions/testing/new_Text.jpeg')

    #test_dataset()





def test_dataset():
    #read models

    print("start")
    with open(database_path+'/models/line_model_predictBinFinal.json', 'r') as f:
        l_model_predict = model_from_json(f.read())
    l_model_predict.load_weights(database_path+'/epochs/Baciu BinFinal--11--1.833.h5')
    # read char list
    f = open(database_path+'/characters.txt', "r")
    string = f.read()
    char_list = []
    char_list[:0] = string
    f=open(database_path+'/datasplitTest.txt', 'r')
    test_data=[]
    for word_img_path in f:
        word,img_path=word_img_path.split(' word-split-path ')
        img_path=img_path[:-1]
        test_data.append(Sample(word,img_path))

    predict = Prediction(64, 128, l_model_predict, char_list, test_data=test_data)
    predict.testing()



if __name__ == '__main__':
    main()


