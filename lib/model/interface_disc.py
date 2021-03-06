# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import model.vertical_disc as vc
import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
import tensorflow as tf
from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob
from nets.vgg16 import vgg16
from model.config import cfg
from model.bbox_transform import clip_boxes, bbox_transform_inv
CLASSES = ('__background__', 'bone', 's')

net = vgg16(batch_size=1)
model = "/Users/anekisei/Documents/tf-faster-rcnn2/output/default/vertical/default/vgg16_faster_rcnn_iter_1500.ckpt"
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
# init session
sess = tf.Session(config=tfconfig)
net.create_architecture(sess, "TEST", len(CLASSES), tag='',
                        anchor_scales=cfg.ANCHOR_SCALES,
                        anchor_ratios=cfg.ANCHOR_RATIOS)
saver = tf.train.Saver()
saver.restore(sess, model)

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
  
  boxes = rois[:, 1:5] / im_scales[0]
  # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def distance(point1,point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    store = []
    vertical_points = []
    '''
    shape[0] -- h -- y, shape[1] -- w -- x
    '''
    for i in range(dets.shape[0]):
        scores = dets[i, -1]
        if scores < thresh:
            continue
        x1 = int(dets[i,0]) # DOUBLE-CHECK THE DIMENSIONS
        y1 = int(dets[i,1])
        x2 = int(dets[i,2])
        y2 = int(dets[i,3])
        area = (y2 - y1)*(x2 - x1)
        center_x = int((x1 + x2) / 2.0)
        center_y = int((y1 + y2) / 2.0)
        vertical_points.append((center_x, center_y))
    return vertical_points

def contrast(img):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(4,4))
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    
    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2
def filter_outliers(vertical_points): #filter out outliers
    points = vertical_points[::-1] #down to up
    i = 0
    j = i + 1
    delete = []
    while i < len(points)-1 and j < len(points):
        lower = points[i]
        upper = points[j]
        if np.abs(lower[0]-upper[0]) >= 50:
            delete.append(upper)
            j += 1
            continue
        i = j
        j += 1
    for item in delete:
        vertical_points.remove(item)
    return vertical_points

def Draw_bone(img, vertical_points):
    vertical_points = vertical_points[::-1] #down to up
    bones = ['S','L5','L4','L3','L2','L1','T12','T11']
    length = min(len(bones), len(vertical_points))
    for i in range(length):
        classification = bones[i]
        center_x,center_y = vertical_points[i]
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 255), -1)
        cv2.putText(img, classification , (center_x - 10, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.CV_AA)
    return img

def test_vertical(img, thresh=0.05, sess=sess):
  """Test a Fast R-CNN network on an image database."""
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  # timers
  im = contrast(img)
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  _t['im_detect'].tic()
  scores, boxes = im_detect(sess, net, im)
  _t['im_detect'].toc()

  _t['misc'].tic()
  # skip j = 0, because it's the background class
  vertical_points = []
  for j, cls in enumerate(CLASSES[1:]):
    j += 1
    inds = np.where(scores[:, j] > thresh)[0]
    cls_scores = scores[inds, j]
    cls_boxes = boxes[inds, j*4:(j+1)*4]
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
       .astype(np.float32, copy=False)
    keep = nms(cls_dets, cfg.TEST.NMS)
    cls_dets = cls_dets[keep, :]
    vertical_points += vis_detections(im, cls, cls_dets, thresh=0.7)
  # sort verticals
  vertical_points = sorted(vertical_points, key=lambda vertical_points: vertical_points[1], reverse=False)#[:7]
  #res_image, flag, disease_bone = vc.spine_contour(im, vertical_points)
  vertical_points = filter_outliers(vertical_points)
  res_image = vc.spine_contour(img, vertical_points)
  res_image = Draw_bone(res_image,vertical_points)
  res2 = vc.spine_contour(im, vertical_points)#contrast img
  res2 = Draw_bone(res2,vertical_points)
  _t['misc'].toc()

  return res_image, res2#, flag, disease_bone
  '''
    if cv2.waitKey(1) & 0xff == 27:
        break
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))
    #for cls_ind, cls in enumerate(CLASSES[1:]):
#vis_detections(im, class_name, dets, thresh=0.5)
  
  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)
  '''
