################Class which build the fully convolutional neural net###########################################################

import inspect
import os
#import TensorflowUtils as utils
import numpy as np
import tensorflow as tf


VGG_MEAN = [103.939, 116.779, 123.68]# Mean value of pixels in R G and B channels

#========================Class for building the FCN neural network based on VGG16==================================================================================
class BUILD_NET_VGG16:
    def __init__(self, vgg16_npy_path=None, is_training=True):
        # if vgg16_npy_path is None:
        #     path = inspect.getfile(BUILD_NET_VGG16)
        #     path = os.path.abspath(os.path.join(path, os.pardir))
        #     path = os.path.join(path, "vgg16.npy")
        #     vgg16_npy_path = path
        #
        #     print(path)
        self.training = is_training
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item() #Load weights of trained VGG16 for encoder
    #print("npy file loaded")
########################################Build Net#####################################################################################################################
    def build(self, rgb):  #Build the fully convolutional neural network (FCN) and load weight for decoder based on trained VGG16 network
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values 0-255
        """
        self.SumWeights = tf.constant(0.0, name="SumFiltersWeights") #Sum of weights of all filters for weight decay loss


#print("build model started")
        # rgb_scaled = rgb * 255.0

        # Convert RGB to BGR and substract pixels mean
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)

        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

#-----------------------------Build network encoder based on VGG16 network and load the trained VGG16 weights-----------------------------------------
       #Layer 1
        self.conv1_1 = self.conv_layer(bgr, "conv1_1") #Build Convolution layer and load weights
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")#Build Convolution layer +Relu and load weights
        self.pool1 = self.max_pool(self.conv1_2, 'pool1') #Max Pooling
       # Layer 2
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # Layer 3
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        # Layer 4
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        # Layer 5
        self.pool4 = tf.stop_gradient(self.pool4)
        
        self.pool4 = tf.contrib.layers.batch_norm(self.pool4)
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_2 = tf.contrib.layers.dropout(self.conv5_2,keep_prob=0.8,is_training=self.training)
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
##-----------------------Build Net classification layer------------------------------------------------------------------------------------
        
        #N x 2 output
        #print("FCN model built")
        self.flatten = tf.reshape(self.pool5, [-1, 2*2*512])
        self.predict = tf.contrib.layers.fully_connected(inputs = self.flatten ,num_outputs = 2, activation_fn=tf.nn.relu)
        
        return self.predict
#####################################################################################################################################################
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
############################################################################################################################################################
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

############################################################################################################################################################
    def conv_layer_NoRelu(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

#########################################Build fully convolutional layer##############################################################################################################
    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
######################################Get VGG filter ############################################################################################################
    def get_conv_filter(self, name):
        var=tf.Variable(self.data_dict[name][0], name="filter_" + name)
        self.SumWeights+=tf.nn.l2_loss(var)
        return var
##################################################################################################################################################
    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases_"+name)
#############################################################################################################################################
    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights_"+name)
