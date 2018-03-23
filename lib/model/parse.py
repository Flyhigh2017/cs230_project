import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
import pandas
#parse label.csv
def parsing(Train=True):
    my_data = pandas.read_csv('/Users/anekisei/Documents/tf-faster-rcnn/lib/model/label.csv').as_matrix()
    store_mp = {}
    if Train:
        start = 0
        end = int(my_data.shape[0])
    else:
        start = int(0.8*my_data.shape[0])
        end = my_data.shape[0]
    for i in range(start,end): #test 0.8, validate 0.2
        annotated = my_data[i][1].split(',')
        if annotated[0] == '0':
            continue
        #process labeled
        for j in range(len(annotated)):
            annotated[j] = int(annotated[j]) - 1
        store_mp[my_data[i][0]+".jpg"] = annotated
            #print store_mp
    return store_mp
#parsing()
