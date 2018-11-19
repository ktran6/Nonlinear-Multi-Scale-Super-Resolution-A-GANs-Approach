#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time, argparser, sys
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
#from dilated_conv_model import *
#from feed_forward_model import *
from gan_utils import *
from config_320 import config, log_config
from random import shuffle

from tensorflow.python.tools import inspect_checkpoint as chkp

###====================== HYPER-PARAMETERS ===========================###

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', help="Either CelebA or WV3", default="WV3")
parser.add_argument('--dataset_path', help="/path/to/data")
parser.add_argument('--batch_size', help="# of images per batch", default=4)
parser.add_argument('--model', help="MSSRGAN, MSSR, or SRGAN", default="MSSRGAN")

args = parser.parse_args()

batch_size = args.batch_size
ni = int(np.sqrt(batch_size))

random.seed(3)

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    #for idx in range(0, len(img_list), n_threads):
    for idx in range(0, img_list, n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))

    #shuffle(imgs)
    return imgs

def evaluate():
    ## create folders to save result images
    save_dir = os.path.join(dataset_path, "/evaluate_{}".format(args.dataset))
    #save_dir = "/project/data/samples/evaluate_multiple_scale_celeb_2"
    #save_dir = "/project/data/samples/evaluate_feed_forward_celeba_2"
    save_lr_dir = os.path.join(dataset_path, "/evaluate_lr_{}".format(args.dataset))
    save_hr_dir = os.path.join(dataset_path, "/evaluate_hr_{}".format(args.dataset))
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir(save_lr_dir)
    tl.files.exists_or_mkdir(save_hr_dir)

    checkpoint_dir = os.path.join(dataset_path, "/checkpoint_{}".format(args.dataset))
    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_img_list = sorted(tl.files.load_file_list(path=os.path.join(args.dataset_path, args.dataset, '/valid_images_hr/'), regx='.*.jpg', printable=False))
    
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=os.path.join(args.dataset_path, args.dataset, n_threads=32)
    valid_hr_imgs = np.array(valid_hr_imgs)

    if args.dataset == "CelebA":
        img_sizes = [16, 128]
    else:
        img_sizes = [40, 320]

    valid_hr_imgs = tl.prepro.threading_data(valid_hr_imgs, fn=crop_sub_imgs_fn, size=img_sizes[1])
    valid_lr_imgs = tl.prepro.threading_data(valid_hr_imgs, fn = downsample_fn, size=img_sizes[0], target_size=img_sizes[1])
    
    #valid_hr_imgs = tl.prepro.threading_data(valid_hr_imgs, fn=crop_sub_imgs_fn, size=320)
    #valid_lr_imgs = tl.prepro.threading_data(valid_hr_imgs, fn = downsample_fn, size=40, target_size=320)

    t_image = tf.placeholder('float32', [batch_size, img_sizes[0], img_sizes[0], 3], name='input_image')
    #t_image = tf.placeholder('float32', [batch_size, 40, 40, 3], name='input_image')
    #net_g = SRGAN_g(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg = 3)
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    init_restorer = tf.train.Saver(g_vars)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    if os.path.isfile(checkpoint_dir+'/gan.ckpt'):
        init_restorer.restore(sess, checkpoint_dir+'/gan.ckpt')
    else:
        init_restorer.restore(sess, checkpoint_dir+'/init.ckpt')

    for idx in range(0, len(valid_lr_imgs), batch_size):
        size = valid_lr_imgs.shape
        valid_hr_batch = valid_hr_imgs[idx:idx + batch_size]
        valid_lr_batch = valid_lr_imgs[idx:idx + batch_size]
        #print(valid_lr_img.shape)
        ###========================== RESTORE G =============================###
        #chkp.print_tensors_in_checkpoint_file(checkpoint_dir+'/gan.ckpt', tensor_name='', all_tensors=True)
        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run([net_g.outputs], {t_image: valid_lr_batch})
        out = out[0]
        print("took: %4.4fs" % (time.time() - start_time))
        print("LR size: %s /  generated HR size: %s" % (size, out.shape))
        print("[*] save images")

        for k in range(batch_size):
            tl.vis.save_image(out[k], save_dir+'/valid_gen_{}.png'.format(idx + k))
            tl.vis.save_image(valid_lr_batch[k], save_lr_dir+'/valid_lr_{}.png'.format(idx + k))
            tl.vis.save_image(valid_hr_batch[k],  save_hr_dir+'/valid_hr_{}.png'.format(idx + k))

if __name__ == '__main__':
    evaluate()
