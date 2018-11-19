#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from dilated_pro_model import *
from gan_utils import *
from config_320 import config, log_config
from random import shuffle

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))
target_size = [40, 80, 160, 320]
ginit_epoch = [100, 100, 100, 100]
gans_epoch = [150, -1, 150, 250]

#ginit_epoch = [0, 0, 0, 0]
#gans_epoch = [0, -1, -1, -1]

random.seed(3)

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))

    shuffle(imgs)
    return imgs

def train(pg, train_hr_imgs):
    ## create folders to save result images and trained model
    save_dir_ginit = "/project/data/samples/{}_mod_more_g_unbound_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "/project/data/samples/{}_mod_more_g_unbound_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "/project/data/checkpoint_mod_more_g_unbound"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, target_size[0], target_size[0], 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, target_size[pg], target_size[pg], 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)
    net_d, _, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)
    _,     _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)

    net_g.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=tf.AUTO_REUSE)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=pg)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss = (tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real))
    #d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    #d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    #d_loss = d_loss1 + d_loss2

    #g_gan_loss = 2e-4 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    g_gan_loss = 5e-3 * -tf.reduce_mean(logits_fake)
    
    mse_loss = .5 * tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    vgg_loss = 5e-7 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = vgg_loss + mse_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    g_old_vars = []
    for i in range(1,pg):
        old_vars = [var for var in g_vars if 'pg{}'.format(i) in var.name]
        g_old_vars = g_old_vars + old_vars

    g_new_vars = [var for var in g_vars if 'pg{}'.format(pg) in var.name]

    #init_restorer = tf.train.Saver(g_new_vars)

    if pg != 1:
        init_restorer = tf.train.Saver(g_old_vars)
    else:
        init_restorer = tf.train.Saver(g_new_vars)

    saver = tf.train.Saver(g_vars + d_vars)
    init_saver = tf.train.Saver(g_vars)


    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1,name='Adam_g_init_{}'.format(pg)).minimize(mse_loss, var_list=g_vars, name='min_g_init_{}'.format(pg))
    ## SRGAN
    #g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    #d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.RMSPropOptimizer(1e-5,name='RMS_g_{}'.format(pg)).minimize(g_loss, var_list=g_new_vars, name='min_g_{}'.format(pg))
    d_optim = tf.train.RMSPropOptimizer(1e-5,name='RMS_d_{}'.format(pg)).minimize(d_loss, var_list=d_vars, name='min_d_{}'.format(pg))

    clip_op = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in d_vars]

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    
    if os.path.isfile(checkpoint_dir+'/gan.ckpt.meta'):
        init_restorer.restore(sess, checkpoint_dir+'/gan.ckpt')
    elif os.path.isfile(checkpoint_dir+'/init.ckpt.meta'):
        init_restorer.restore(sess, checkpoint_dir+'/init.ckpt')

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "../vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training

    train_lr_imgs = tl.prepro.threading_data(train_hr_imgs, fn=crop_sub_imgs_fn, size=target_size[pg])
    train_hr_imgs = tl.prepro.threading_data(train_hr_imgs, fn=crop_sub_imgs_fn, size=target_size[3])
    sample_imgs = train_lr_imgs[0 : batch_size]
    sample_imgs_target = sample_imgs[0 : batch_size]
    sample_imgs_start = tl.prepro.threading_data(train_hr_imgs[0 : batch_size], fn=downsample_fn, size=target_size[0], target_size=target_size[3])
    
    tl.vis.save_images(sample_imgs_start, [ni, ni], save_dir_ginit+'/_train_sample_{}.png'.format(target_size[0]))
    tl.vis.save_images(sample_imgs_target, [ni, ni], save_dir_ginit+'/_train_sample_{}.png'.format(target_size[pg]))
    tl.vis.save_images(sample_imgs_start, [ni, ni], save_dir_gan+'/_train_sample_{}.png'.format(target_size[0]))
    tl.vis.save_images(sample_imgs_target, [ni, ni], save_dir_gan+'/_train_sample_{}.png'.format(target_size[pg]))
    
    print('done saving samples')
    
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, ginit_epoch[pg]+1):
        #break
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_target = train_lr_imgs[idx : idx + batch_size]
            b_imgs_start = tl.prepro.threading_data(train_hr_imgs[idx : idx + batch_size], fn=downsample_fn, size = target_size[0], target_size=target_size[3])
            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_start, t_target_image: b_imgs_target})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, ginit_epoch[pg], n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, ginit_epoch[pg], time.time() - epoch_time, total_mse_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out, errM = sess.run([net_g_test.outputs, mse_loss], {t_image: sample_imgs_start, t_target_image: b_imgs_target})
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_{}_{}_output.png'.format(pg,epoch))

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            saver.save(sess, checkpoint_dir+'/init.ckpt')

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, gans_epoch[pg] + 1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, g_iter, d_iter = 0, 0, 0, 0

        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_target = train_lr_imgs[idx : idx + batch_size]
            b_imgs_start = tl.prepro.threading_data(train_hr_imgs[idx : idx + batch_size], fn=downsample_fn, size = target_size[0], target_size=target_size[3])

            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_start, t_target_image: b_imgs_target})
            total_d_loss += errD
            d_iter += 1
            sess.run(clip_op)

            ## update G
            if idx % 16 == 0:
                errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_start, t_target_image: b_imgs_target})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, gans_epoch[pg], g_iter, time.time() - step_time, errD, errG, errM, errV, errA))
                total_g_loss += errG
                g_iter += 1

                log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, gans_epoch[pg], time.time() - epoch_time, total_d_loss/g_iter, total_g_loss/g_iter)
                print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_start})
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_{}_{}.png'.format(pg,epoch))

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            saver.save(sess, checkpoint_dir+'/gan.ckpt')

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = 64 # 0: 企鹅  81: 蝴蝶 53: 鸟  128: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir+'/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir+'/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir+'/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir+'/valid_bicubic.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')
    
    args = parser.parse_args()

    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.jpg', printable=False))
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        for i in range(1,4):
            train(i, train_hr_imgs)
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
