# Run prediction and genertae pixelwise annotation for every pixels in the image using fully coonvolutional neural net
# Output saved as label images, and label image overlay on the original image
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input image for prediction are located
# 3) Set number of classes number in NUM_CLASSES
# 4) Set Pred_Dir the folder where you want the output annotated images to be save
# 5) Run script
#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import sys
import BuildNetVgg16
from PIL import Image
import os
import cv2
'''
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
Image_Dir="/Users/anekisei/Documents/Spine_project/test_images"# Test image folder
w=0.6# weight of overlay on image
Pred_Dir="/Users/anekisei/Documents/Spine_project/FCN_segment/output/" # Library where the output prediction will be written
'''
#model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
'''
NameEnd="" # Add this string to the ending of the file name optional
NUM_CLASSES = 3 # Number of classes
'''
#-------------------------------------------------------------------------------------------------------------------------
#CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it

################################################################################################################################################################################
def softmax(x):
    res = np.zeros(x.shape)
    res[:,0] = np.exp(x[:,0]) / np.sum(np.exp(x), axis=1)
    res[:,1] = np.exp(x[:,1]) / np.sum(np.exp(x), axis=1)
    return res
def predict(img):
    tf.reset_default_graph()
    logs_dir= "/Users/anekisei/Documents/vertical_crop/logs/"# "path to logs directory where trained model and information will be stored"
    Image_Dir="/Users/anekisei/Documents/vertical_crop/train_images/"# Test image folder
    model_path="/Users/anekisei/Documents/vertical_crop/Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB

    # -------------------------Build Net----------------------------------------------------------------------------------------------
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path,is_training=False)  # Create class instance for the net
    logits = Net.build(image)
    sess = tf.Session() #Start Tensorflow session
    sess.run(tf.global_variables_initializer())
    #print("Setting up Saver...")
    saver = tf.train.Saver()
    saver.restore(sess, "/Users/anekisei/Documents/vertical_crop/logs/model.ckpt-500")
    print "restore 500"
    '''
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        #print "Restore from:", ckpt.model_checkpoint_path
    #saver.restore(sess, "/Users/anekisei/Documents/vertical_crop/logs/model.ckpt-2000")
        saver.restore(sess, ckpt.model_checkpoint_path)
    #print("Model restored...")
    else:
        print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
        sys.exit()
    '''
    feed_dict = {image:img}
    logits = sess.run(logits, feed_dict=feed_dict)
    logits = softmax(logits)
    '''
    results = []
    for i in range(logits.shape[0]):
        if logits[i,1] <= 0.8:
            print "healthy"
            results.append(True)#"health"
        else:
            print "disease"
            results.append(False)#"disease"
    sess.close()
    return results
    '''
    return logits
#predict()#Run script
#print("Finished")
