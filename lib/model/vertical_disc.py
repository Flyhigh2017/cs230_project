import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
import pdb
from matplotlib import pyplot as plt
import Crop_disc
import parse as par
import Inference as Inf
from label_image import run_inference
#import classifier.Inference as Judge
def spine_contour(img, vertical_points): #up to down
    bones_rev = ['S','L5','L4','L3','L2','L1','T12','T11','Unknown','Unknown']
    bones = bones_rev[0:len(vertical_points)][::-1]
    index = len(vertical_points)-1
    #crop and rotate inbetween zones, need an labeled array
    store = []
    #correct = 0
    flag = False
    #tensor = np.zeros((len(vertical_points)-1,60,60,3))
    img_list = []
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(len(vertical_points)-1):
        '''
        if i in labeled:
            tag = "d"
        else:
            tag = "h"
        '''
        point1 = vertical_points[i]
        point2 = vertical_points[i+1]
        if i == len(vertical_points)-2:
            point1 = (point1[0]-5, point1[1]+5)
            point2 = (point2[0]-5, point2[1]+5)
        if i in [len(vertical_points)-2, len(vertical_points)-3]:
            out, center, angle = Crop_disc.find_disc(gray_img, point1, point2, height_constant=0.85,width_constant=1.5)
        else:
            out, center, angle = Crop_disc.find_disc(gray_img, point1, point2)
        #rotate
        M = cv2.getRotationMatrix2D(center,-angle,1)
        res = cv2.warpAffine(out,M,(gray_img.shape[1],gray_img.shape[0]))
        
        #cropped = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR) #res
        contour_out, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#with angle
        cv2.drawContours(img,contour_out,-1,(0,255,0),1)
        #contour_rect, hierarchy_rect = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#res, rotated
        cv2.putText(img, str(index), (center[0]-20,center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.CV_AA)
        
        index -= 1
        '''
        (x, y, w, h) = cv2.boundingRect(contour_rect[0])
        cropped = cropped[y:y+h,x:x+w]
        store.append((contour_out,point1))
        img_list.append(cropped)
        
        resized = cv2.resize(cropped, (60, 60))
        tensor[i,:,:,:] = resized
        '''
    '''
    disease_bone = []
    results1 = run_inference(img_list)
    results2 = Inf.predict(tensor)
    for i in range(results1.shape[0]):
        contour_out, point1 = store[i]
        if (results1[i,0] + results2[i,0]) / 2.0 <= 0.4: #disease
            flag = True
            disease_sec = bones[i] + '-' + bones[i+1]
            disease_bone.append(disease_sec)
            cv2.drawContours(img,contour_out,-1,(0,0,255),1)
            cv2.putText(img, "disease", (point1[0]+10, point1[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.CV_AA)
    #disease_bone = list(set(disease_bone))
    '''
    return img#, flag, disease_bone


