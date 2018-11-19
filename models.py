#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py

def MSSRGAN_g(t_image, is_train=False, reuse=False, pg=3):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        concatInput = []

        for i in range(1, pg + 1):
            if i < pg: 
                concatInput.append(n)

            convConcatInput = []
            for f in range(i-1):
                convInput = concatInput[f]
                convInput = DeConv2d(convInput, 2 ** (i + 4), (2 ** (i - f - 1), 2 ** (i - f - 1)), strides=(2 ** (i - f - 1), 2 ** (i - f - 1)), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n12/pg{}/input_concat/stage{}'.format(i,f))
                convConcatInput.append(convInput)

            n = Conv2d(n, 2 ** (i+5), (3, 3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='init_conv/pg{}'.format(i))

            temp = n
            
            if len(convConcatInput) > 0:
                convConcatInput.append(n)
                n = ConcatLayer(convConcatInput, concat_dim=3, name='concatInput/pg{}'.format(i))
            
            concatList = []
            for j in range(1, i + 3):
                nAC = AtrousConv2dLayer(n, 8 * (i + 9), (3, 3), j, act=tf.nn.relu, padding='SAME', W_init=w_init, name='n8/pg{}/init/ac{}'.format(i,j))
                concatList.append(nAC)
            n = ConcatLayer(concatList, concat_dim=3, name='res_cat/init/pg{}'.format(i))
            n = Conv2d(n, 8 * (i + 9) * (i + 2), (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res_dil_conv_mod/init/pg{}'.format(i))

            # B residual blocks
            for k in range(2 ** (5-i)):
                concatList = []
                for j in range(1, i + 3):
                    nAC = AtrousConv2dLayer(n, 8 * (i + 9), (3, 3), j, act=tf.nn.relu, padding='SAME', W_init=w_init, name='n8/pg{}/res{}/ac{}'.format(i,k,j))
                    concatList.append(nAC)
                nn = ConcatLayer(concatList, concat_dim=3, name='res_cat/res{}/pg{}'.format(k,i))
                nn = Conv2d(nn, 8 * (i + 9) * (i + 2), (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res_dil_conv_mod/res{}/pg{}'.format(k,i))
                nn = ElementwiseLayer([n, nn], tf.add, name='res_add/res{}/pg{}'.format(k,i))
                n = nn

            n = Conv2d(n, 2 ** (i+5), (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='post_conv/pg{}'.format(i))
            n = ElementwiseLayer([n, temp], tf.add, name='add3/pg{}'.format(i))
            # B residual blacks end

            n = Conv2d(n, 2 ** (i + 6), (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='pre_up/pg{}'.format(i))
            n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/pg{}'.format(i))

            n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out/pg{}'.format(i))
        return n


def MSSRGAN_d(input_images, is_train=True, reuse=False, pg=3):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 2 ** pg * 16
    lrelu = lambda x: tl.act.lrelu(x, 0.0)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        net_in = InputLayer(input_images, name='input/images')

        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='h0/c_{}'.format(pg))

        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c_{}'.format(pg))
        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c_{}'.format(pg))
        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c_{}'.format(pg))

        net = Conv2d(net_h3, df_dim * 2, (1, 1), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='res/c_{}'.format(pg))
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='res/c2_{}'.format(pg))
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c3_{}'.format(pg))
        net_h4 = ElementwiseLayer([net_h3, net], combine_fn=tf.add, name='res/add_{}'.format(pg))

        net_ho = FlattenLayer(net_h4, name='ho/flatten_{}'.format(pg))
        #net_ho = DenseLayer(net_ho, n_units=1024, act=lrelu, W_init = w_init, name='ho/d1024{}'.format(pg))
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init = w_init, name='ho/d1_{}'.format(pg))
        logits = net_ho.outputs
        logits = tf.nn.tanh(net_ho.outputs)

    return net_ho, net_ho.outputs, logits

def SRGAN_g(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/3')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/3')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n

def SRGAN_d(t_image, is_train=False, reuse=False):
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits

def MSSR_g(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = UpSampling2dLayer(n, size=(8,8), is_scale=True, method=2, name='upsample2d_layer')
        temp = n

        concatList = []
        for j in range(1,4):
            nAC = AtrousConv2dLayer(n, 8, (3, 3), j, act=tf.nn.relu, padding='SAME', W_init=w_init, name='n8/init/ac{}'.format(j))
            concatList.append(nAC)
        n = ConcatLayer(concatList, concat_dim=3, name='cat/init/128')
        n = Conv2d(n, 24, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='res_dil_conv_mod/init')

        # B residual blocks
        for k in range(4):
            concatList = []
            for j in range(1,4):
                nAC = AtrousConv2dLayer(n, 8, (3,3), j, act=tf.nn.relu, padding='SAME', W_init=w_init, name='n8/res{}/ac{}'.format(k,j))
                concatList.append(nAC)
            nn = ConcatLayer(concatList, concat_dim=3, name='res_cat/res{}{}'.format(k,j))
            nn = Conv2d(nn, 24, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='res_dil_conv_mod/res{}'.format(k))
            n = nn

        n = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='post_conv')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        #n = Conv2d(n, 3, (3,3), (1,1) act=None, padding='SAME', W_init=w_init, b_init=b_init, name='final_conv/pg{}'.format(i))
    return n

def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool2')
        conv = network
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool3')
        #conv = network
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool4')                               # (batch_size, 14, 14, 512)
        #conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool5')                               # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv
