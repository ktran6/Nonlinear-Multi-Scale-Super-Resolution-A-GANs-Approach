import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
import cv2
import random

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, size):
    #x = crop(x, wrg=600, hrg=600, is_random=is_random)
    x = cv2.resize(x, dsize=(size,size), interpolation=cv2.INTER_CUBIC)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_keep_ratio(x):
    x = x / (255. / 2.)
    x = x - 1.
    height = x.shape[0]
    width = x.shape[1]
    return cv2.resize(x, (int(width/8), int(height/8)))

def downsample_fn(x,size, target_size):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    i = random.randint(0,1)
    k = random.randint(0,4)
    x = (x + 1) * (255. / 2.)
    if i != 0:
        x = x + np.random.uniform(0,12,(target_size,target_size,3))
        x = x.clip(min = 0, max = 255)
    x = cv2.resize(x, dsize=(size, size), interpolation=k) 
    x = x / (255. / 2.) 
    x = x - 1.
    return x

def upsample_fn(x,size):
   types = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
   i = random.randint(0,2)
   x = (x + 1) * (255. / 2.)
   x = cv2.resize(x, dsize=(size, size), interpolation=types[1])
   #x = x + np.random.uniform(0,12,(150,150,3))
   #x = x.clip(min=0, max=255)
   x = x / (255. / 2.) - 1
   return x
