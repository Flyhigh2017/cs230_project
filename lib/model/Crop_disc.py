import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
#imagepath = "/Users/anekisei/Documents/bulge/1108304611.jpg"
#maskpath = "./Label/1_1_9.png"
#img = cv2.imread(imagepath,0)



def find_disc(img, point1, point2, height_constant=0.7, width_constant=1.3):
    #img is grayscale
    mask = np.zeros_like(img)
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    d = np.sqrt((y2 - y1)*(y2 - y1) + (x2 - x1)*(x2 - x1))
    center = (int((x1+x2)/2.0), int((y1+y2)/2.0))
    height = int(height_constant * d)
    width = int(width_constant * d)
    top_left = (int(center[0] - width/2.0),int(center[1] - height/2.0))
    bottom_right = (int(center[0] + width/2.0),int(center[1] + height/2.0))
    cv2.rectangle(mask, top_left, bottom_right, 255, 2)

    #rotate
    angle = np.arctan2((point2[0] - point1[0] + 0.0), (point2[1] - point1[1] + 0.0)) * 180 / np.pi
    M = cv2.getRotationMatrix2D(center,angle,1)
    mask = cv2.warpAffine(mask,M,(mask.shape[1],mask.shape[0]))

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, 0, 255, -1) # Draw filled contour in mask
    
    out = np.zeros_like(img) # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]
    #cv2.line(out, point1, point2, 255, 3)
    #cv2.imshow('image',out)
    #cv2.waitKey(0)
    return out, center, angle

'''
point1 = (100,100)
point2 = (80,200)
find_disc(img,point1,point2)
'''



