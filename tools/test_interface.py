# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test_vertical import test_net_vertical
from model.test_horizontal import test_net_horizontal
from model.test_damage import test_net_damage
from model.interface_disc import test_vertical
import model.vertical_crop
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import cv2
import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from os.path import isfile, join
from os import listdir
def test_v(im):
    return test_vertical(im)

data_path = "/Users/anekisei/Documents/tf-faster-rcnn2/tools/tj"
def transfer(index):
    string = str(index)
    if len(string) == 1:
        return '00'+string
    if len(string) == 2:
        return '0'+string
    if len(string) == 3:
        return string
i = 1
for patient in listdir(data_path): #patient = TJ-00X
    print (patient)
    if patient == '.DS_Store':
        continue
    #if patient != 'TJ-019':
    #continue
    #if i < 17:
    # i += 1
    #continue
    type = join(data_path, patient)
    for temp in listdir(type):
        if temp == '.DS_Store':
            continue
        if temp == 'H':
            continue
        direct = join(type,temp)  #temp = H, T1, T2
        for filename in listdir(direct):
            if filename == '.DS_Store':
                continue
            print (filename)
            img = cv2.imread(direct + "/" + filename)
            res1, res2 = test_v(img)
            cv2.imshow('image',res1)
            cv2.waitKey(1000)
            cv2.imwrite("./save1/" + filename,res1)
            cv2.imwrite("./save2/" + filename,res2)
            #os.rename(direct + "/" + filename, direct + "/" + patient + "-" + temp + "-" + transfer(index) + ".jpg")
