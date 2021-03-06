from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
import copy
from os import listdir
from os.path import isfile, join
class vertical(imdb):
    def __init__(self, isFinish):
        imdb.__init__(self, 'vertical')
        self._data_path = '/Users/anekisei/Documents/tf-faster-rcnn/data/vertical'
        self.label_path = '/Users/anekisei/Documents/tf-faster-rcnn/data/vertical/Annotations'
        self.image_path = join(self._data_path, 'JPEGImages')
        self.label_matrix = np.loadtxt(os.path.join(self.label_path, 'label.txt'),dtype='str')
        self.file_names = [ f for f in listdir(self.image_path) if isfile(join(self.image_path,f)) ]
        self._classes = {'__background__': '0', 'bone': '1', 's': '2'}
        self._class_to_ind = dict(list(zip(self._classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self.finish = isFinish
        self.image_count = 0
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])
                
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',index)
        assert os.path.exists(image_path), \
          'Path does not exist: {}'.format(image_path)
        return image_path
    
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        return self.file_names


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        
        This function loads/saves from/to a cache file to speed up future calls.
        """

        gt_roidb = [self._load_vertical_annotation(index)
                    for index in self._image_index]

        return gt_roidb
    '''
    def genrate_matrix(self,im_list):
        N = len(im_list)
        h, w, channel = im_list[0].shape
        X_train = np.zeros(shape=(N,h,w,channel))
        for i in range(N):
            X_train[i,:,:,:] =im_list[i]
        return X_train


    def image_read(self, mypath, width, height, batch_start):#modify to list[matrix] in the future width=1024
        onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
        images = []
        start_index = batch_start * self.batch_size
        for n in range(start_index, start_index + self.batch_size):
            if (onlyfiles[n][-1] == 'g') and (n < len(onlyfiles)):
                raw_image = cv2.imread( join(mypath,onlyfiles[n]) )
                resized_image = cv2.resize(raw_image, (width, height))
                imag = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32)
                imag = (imag / 255.0) * 2.0 - 1.0
                images.append(imag)
    #dataset = self.genrate_matrix(images)
        return images

    def get_batch(self):
        images_list = self.image_read(self.image_path, self.image_width, self.image_height, self.batch_start)
        images_matrix = self.genrate_matrix(images_list)
        #label_batch_list = self.label_list[self.batch_start * self.batch_size : self.batch_start * self.batch_size + self.batch_size]
        label_batch_list = self.label_load_batch(self.label_path, self.class_dic, self.batch_start)
        labels, object_mask = self.label_transfer_mask(images_matrix, label_batch_list, self.cell_size1, self.cell_size2, self.num_class)
        self.batch_start += 1
        if self.batch_size * (self.batch_start + 1) > len(self.file_names):
            self.finish = True
        return images_matrix, labels, object_mask


    '''
    def _load_vertical_annotation(self, index):#modify to list[matrix] in the future
        print ("image is,", index, self.image_count)
        self.image_count = self.image_count + 1
        label_matrix = np.delete(self.label_matrix,[0],0)
        #xmin,xmax,ymin,ymax,Frame,Label,Preview URL
        '''
        center_x = ((leftTop_x + rightBot_x) / 2).astype(np.int)
        center_y = ((leftTop_y + rightBot_y) / 2).astype(np.int)
        width = np.absolute(rightBot_x - leftTop_x)
        height = np.absolute(rightBot_y - leftTop_y)
        '''
        count = 0
        index_list = []
        for i in range(label_matrix.shape[0]):
            img_name = label_matrix[i,4]
            img_name = img_name[:-3] + 'jpg'
            if(img_name == index):
                count = count + 1
                index_list.append(i)
        boxes = np.zeros((count, 4), dtype=np.uint16)
        gt_classes = np.zeros((count), dtype=np.int32)
        overlaps = np.zeros((count, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((count), dtype=np.float32)

        for i in range(count):
            leftTop_x = int(label_matrix[index_list[i],0])
            leftTop_y = int(label_matrix[index_list[i],1])
            rightBot_x = int(label_matrix[index_list[i],2])
            rightBot_y = int(label_matrix[index_list[i],3])
            type_vec = self._classes[label_matrix[index_list[i],5]]
            type_vec = int(type_vec)
            boxes[i,:] = [leftTop_x, leftTop_y, rightBot_x, rightBot_y]
            gt_classes[i] = type_vec
            overlaps[i,type_vec] = 1.0
            # "Seg" area for pascal is just the box area
            seg_areas[i] = (rightBot_x - leftTop_x + 1) * (rightBot_y - leftTop_y + 1)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}















