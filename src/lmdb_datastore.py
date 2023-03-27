import argparse
import pickle

import cv2
import lmdb
from path import Path



# 2GB is enough for IAM dataset
env = lmdb.open('D:\school-projects\year3sem1\licenta\Iam-dataset\lmdb', map_size=1024 * 1024 * 1024 * 2)

# go over all png files
fn_imgs = list(('D:\school-projects\year3sem1\licenta\Iam-dataset\img').walkfiles('*.png'))

# and put the imgs into lmdb as pickled grayscale imgs
with env.begin(write=True) as txn:
    for i, fn_img in enumerate(fn_imgs):
        print(i, len(fn_imgs))
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        basename = fn_img.basename()
        txn.put(basename.encode("ascii"), pickle.dumps(img))

env.close()
