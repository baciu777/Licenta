import glob
import os
import pickle
import random
from collections import namedtuple


import cv2
import lmdb
import numpy as np
from path import Path


Sample = namedtuple('Sample', 'word, file_path')
Batch = namedtuple('Batch', 'imgs, words, batch_size')


class DataLoaderIAM:
    """
    data loader for dataset IAM - words
    """

    def __init__(self,data_dir: Path,data_split) -> None:

        assert data_dir.exists()
        self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)
        self.data_split=data_split
        self.curr_id = 0
        self.samples = []
        self.train_samples=[]
        self.validation_samples=[]
        self.test_samples=[]

        txt_lines = []



        words = open(data_dir / 'gt/words.txt')
        for line in words:
            if line[0] == "#":
                continue
            if len(line.split(" ")[8]) > 16:
                continue
            if line.split(" ")[1] != "err":  # We don't need to deal with error entries.
                txt_lines.append(line)

        print(len(txt_lines))

        np.random.shuffle(txt_lines)
        #split train, validation, test
        split_id_1 = int(self.data_split[0] * len(txt_lines))
        split_id_2=int(self.data_split[1] * len(txt_lines))
        train_samples = txt_lines[:split_id_1]
        validation_samples = txt_lines[split_id_1:split_id_2]
        test_samples=txt_lines[split_id_2:]





        print(f"Total training samples: {len(train_samples)}")
        print(f"Total validation samples: {len(validation_samples)}")
        print(f"Total test samples: {len(test_samples)}")

        assert len(txt_lines) == len(train_samples) + len(validation_samples) + len(test_samples)

        #characters found in data
        characters = set()
        characters = self.get_image_paths_and_labels(data_dir,train_samples,characters,'train')
        characters = self.get_image_paths_and_labels(data_dir,validation_samples, characters,'validation')
        characters = self.get_image_paths_and_labels(data_dir,validation_samples, characters,'test')


        f = open("../model/testDataSplit.txt", "w")
        for word,path in self.test_samples:
            f.write(str(word)+" word-split-path "+str(path)+"\n")
        f.close()




        # list of all characters in dataset
        self.char_list = sorted(list(characters))

        f = open("../model/list.txt", "w")############
        for el in self.char_list:
            f.write(el)
        f.close()



    def get_image_paths_and_labels(self,data_dir,samples,characters,type_samples):

        for (i, file_line) in enumerate(samples):
            line_split = file_line.strip()
            line_split = line_split.split(" ")

            # Each line split will have this format for the corresponding image:
            # part1/part1-part2/part1-part2-part3.png
            image_name = line_split[0]
            partI = image_name.split("-")[0]
            partII = image_name.split("-")[1]
            img_path = os.path.join(
                data_dir,'img', partI, partI + "-" + partII, image_name + ".png"
            )
           # the text is starting at position 9
            words=line_split[8:]

            word = ' '.join(words)
            characters = characters.union(set(list(word)))

            # put sample into list
            self.samples.append(Sample(word, img_path))
            if type_samples is 'train':
                self.train_samples.append(Sample(word, img_path))
            elif type_samples is 'validation':
                self.validation_samples.append(Sample(word, img_path))
            else:
                self.test_samples.append(Sample(word,img_path))

        return characters

