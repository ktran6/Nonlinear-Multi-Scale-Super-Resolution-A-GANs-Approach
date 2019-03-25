#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time, argparse, sys
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from models import *
from utils import *
from random import shuffle

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', help="Either CelebA or WV3", default="WV3")
parser.add_argument('--dataset_path', help="/path/to/data")
parser.add_argument('--batch_size', help="# of images per batch", default=4)
parser.add_argument('--lr', help="Learning Rate", default=.00001)
parser.add_argument('--model', help="MSSRGAN, MSSR, or SRGAN", default="MSSRGAN")
parser.add_argument('--checkpoint')
parser.add_argument('--texture_only', action='store_true')

args = parser.parse_args()

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = args.batch_size
lr_init = args.lr
beta1 = 0.9 
## initialize G
## adversarial learning (SRGAN)
lr_decay = 0.1

if args.dataset == "WV3":
    target_size = [40, 80, 160, 320]
    ginit_epoch = [100, 100, 100, 100]
    gans_epoch = [150, -1, 150, 250]
    loss_parameters = []

else:
    target_size = [16, 32, 64, 128]
    ginit_epoch = [20, 40, 40, 40]
    gans_epoch = [150, -1, -1, 70]
    loss_parameters = []

if args.model == "MSSR":
    ginit_epoch = [0, 0, 0, 350]

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
    save_dir_model_init = os.path.join(args.dataset_path, "train_init_{}_{}".format(args.dataset,args.checkpoint))
    save_dir_model = os.path.join(args.dataset_path, "train_{}_{}".format(args.dataset,args.checkpoint))
    tl.files.exists_or_mkdir(save_dir_model_init)
    tl.files.exists_or_mkdir(save_dir_model)
    checkpoint_dir = os.path.join(args.dataset_path, "checkpoint_{}_{}".format(args.dataset,args.checkpoint))
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, target_size[0], target_size[0], 3], name='t_image_input_to_generator')
    t_target_image = tf.placeholder('float32', [batch_size, target_size[pg], target_size[pg], 3], name='t_target_image')

    if args.model == "MSSRGAN":
         net_g = MSSRGAN_g(t_image, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)
         net_d, _, logits_real = MSSRGAN_d(t_target_image, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)
         _,     _, logits_fake = MSSRGAN_d(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)

    elif args.model == "MSSRGAN_texture":
        net_g, _ = MSSRGAN_texture_g_2(t_image, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)
        net_d, _, logits_real = MSSRGAN_d(t_target_image, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)
        _, _, logits_fake = MSSRGAN_d(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)

    elif args.model == "MSSRGAN_derivative":
        net_g, _ = MSSRGAN_texture_g_derivative(t_image, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)
        net_d, _, logits_real = MSSRGAN_d(t_target_image, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)
        _, _, logits_fake = MSSRGAN_d(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE, pg=pg)

    elif args.model == "SRGAN":
         net_g = SRGAN_g(t_image, is_train=True)
         net_d, _, logits_real = SRGAN_d(t_target_image, is_train=True)
         _,     _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE)

    elif args.model == "MSSR":
         net_g = MSSR_G(t_image, is_train=True)

    net_g.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=tf.AUTO_REUSE)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
    if args.model == "MSSRGAN":
         net_g_test = MSSRGAN_g(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=pg)

    elif args.model == "MSSRGAN_texture":
        net_g_test, _ = MSSRGAN_texture_g_2(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=pg)
 
    elif args.model == "MSSRGAN_derivative":
        net_g_test, _ = MSSRGAN_texture_g_derivative(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=pg)

    elif args.model == "SRGAN":
         net_g_test = SRGAN_g(t_image, is_train=False, reuse=tf.AUTO_REUSE)

    elif args.model == "MSSR":
         net_g_test = MSSR_G(t_image, is_train=False, reuse=tf.AUTO_REUSE)

    # ###========================== DEFINE TRAIN OPS ==========================###i
    if "MSSRGAN" in args.model:
        d_loss = (tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real))
        g_gan_loss = 7.5e-3 * -tf.reduce_mean(logits_fake)
    
    elif args.model == "SRGAN":
        d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
        d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
        d_loss = d_loss1 + d_loss2
        g_gan_loss = 2e-4 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    
    #mse_loss = .5 * tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    mse_loss = .3 * tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    vgg_loss = 5e-7 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    if args.model != "MSSR":
        g_loss = mse_loss
    else:
        g_loss = vgg_loss + mse_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    if "MSSRGAN" in args.model:
        g_old_vars = []
        for i in range(1,pg):
        #for i in range(1, pg + 1): #Use this line to only train GANs
            old_vars = [var for var in g_vars if 'pg{}'.format(i) in var.name]
            g_old_vars = g_old_vars + old_vars

        g_new_vars = [var for var in g_vars if 'pg{}'.format(pg) in var.name]

    #init_restorer = tf.train.Saver(g_new_vars)
    
    if pg != 1 and "MSSRGAN" in args.model:
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
    if args.model == "SRGAN":
        g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    elif "MSSRGAN" in args.model:
        g_optim = tf.train.RMSPropOptimizer(1e-5,name='RMS_g_{}'.format(pg)).minimize(g_loss, var_list=g_new_vars, name='min_g_{}'.format(pg))
        d_optim = tf.train.RMSPropOptimizer(1e-5,name='RMS_d_{}'.format(pg)).minimize(d_loss, var_list=d_vars, name='min_d_{}'.format(pg))
        clip_op = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in d_vars]

    ###========================== RESTORE MODEL =============================###
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
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
    sample_imgs = train_lr_imgs[1000 : 1000 + batch_size]
    sample_imgs_target = sample_imgs[0 : batch_size]
    sample_imgs_start = tl.prepro.threading_data(train_hr_imgs[1000 : 1000 + batch_size], fn=downsample_fn, size=target_size[0], target_size=target_size[3])
    for img_idx in range(sample_imgs_start.shape[0]):    
        tl.vis.save_image(sample_imgs_start[img_idx], save_dir_model_init+'/_train_sample_{}_{}.png'.format(target_size[0], img_idx))
        tl.vis.save_image(sample_imgs_target[img_idx], save_dir_model_init+'/_train_sample_{}_{}.png'.format(target_size[pg], img_idx))
        tl.vis.save_image(sample_imgs_start[img_idx], save_dir_model+'/_train_sample_{}_{}.png'.format(target_size[0], img_idx))
        tl.vis.save_image(sample_imgs_target[img_idx], save_dir_model+'/_train_sample_{}_{}.png'.format(target_size[pg], img_idx))
    
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
        if (epoch != 0) and (epoch % 5 == 0):
            out, errM = sess.run([net_g_test.outputs, mse_loss], {t_image: sample_imgs_start, t_target_image: b_imgs_target})
            print("[*] save images")
            for img_idx in range(out.shape[0]):
                tl.vis.save_image(out[img_idx], save_dir_model_init+'/train_{}_{}_{}.png'.format(pg,epoch,img_idx))

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            saver.save(sess, checkpoint_dir+'/init.ckpt')

    ###========================= train GAN (SRGAN) =========================###
    decay_every = gans_epoch[pg] / 2
    for epoch in range(0, gans_epoch[pg] + 1):
        if args.model == "MSSR":
            break
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
            if "MSSRGAN" in args.model:
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
        if (epoch != 0) and (epoch % 5 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_start})
            print("[*] save images")
            for img_idx in range(out.shape[0]):
                tl.vis.save_image(out[img_idx], save_dir_model+'/train_{}_{}_{}.png'.format(pg,epoch,img_idx))

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            saver.save(sess, checkpoint_dir+'/gan.ckpt')


def train_texture(train_hr_imgs):
    ## create folders to save result images and trained model
    save_dir_model = os.path.join(args.dataset_path, "train_texture_no_threshold_{}_{}".format(args.dataset,args.checkpoint))
    tl.files.exists_or_mkdir(save_dir_model)
    checkpoint_dir = os.path.join(args.dataset_path, "checkpoint_{}_{}".format(args.dataset,args.checkpoint))
    tl.files.exists_or_mkdir(checkpoint_dir)


    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, target_size[0], target_size[0], 3],
                             name='t_image_input_to_generator')
    t_target_image = tf.placeholder('float32', [batch_size, target_size[3], target_size[3], 3], name='t_target_image')

    if args.model == "MSSRGAN_texture":
        net_g, _ = MSSRGAN_texture_g_2(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=3)
        net_texture = MS_texture(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE)

    ## test inference
        net_g_test, _ = MSSRGAN_texture_g_2(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=3)
        net_texture_test = MS_texture(net_g_test.outputs, is_train=False, reuse=tf.AUTO_REUSE)

    elif args.model == "MSSRGAN_derivative":
        net_g, _ = MSSRGAN_texture_g_derivative(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=3)
        net_texture = MS_texture(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE)

    ## test inference
        net_g_test, _ = MSSRGAN_texture_g_derivative(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=3)
        net_texture_test = MS_texture(net_g_test.outputs, is_train=False, reuse=tf.AUTO_REUSE)
     
    ####========================== DEFINE TRAIN OPS ==========================###i
    #train_output = (net_texture.outputs - tf.reduce_min(net_texture.outputs))/(tf.reduce_max(net_texture.outputs) - tf.reduce_min(net_texture.outputs)) * 2 - 1
    train_output = net_texture.outputs
    diff = t_target_image - net_g.outputs
    diff_positive = tf.keras.activations.relu(diff, threshold=.02)
    diff_negative = tf.multiply(tf.keras.activations.relu(tf.multiply(diff, -1), threshold=.02), -1)

    diff = tf.add(diff_negative, diff_positive)
    mse_mask = tf.add(tf.multiply(tf.cast(tf.equal(diff, 0.), tf.float32), 2), 1)
    penalized_output = tf.multiply(mse_mask, train_output)
    texture_loss = tf.losses.mean_squared_error(penalized_output, diff)
    
    diff_grad = tf.image.image_gradients(tf.image.rgb_to_grayscale(diff))
    diff_grad = tf.stack([diff_grad[0], diff_grad[1]], axis = 3)
    output_grad = tf.image.image_gradients(tf.image.rgb_to_grayscale(train_output))
    output_grad = tf.stack([output_grad[0], output_grad[1]], axis = 3)
    gradient_loss = tf.losses.mean_squared_error(output_grad, diff_grad)
 
    total_loss = texture_loss #+ gradient_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    texture_vars = tl.layers.get_variables_with_name('texture', True, True)

    g_old_vars = []
    for i in range(1, 4):
        old_vars = [var for var in g_vars if 'pg{}'.format(i) in var.name]
        g_old_vars = g_old_vars + old_vars

    init_restorer = tf.train.Saver(g_old_vars)

    saver = tf.train.Saver(texture_vars)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    texture_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, name='texture_optimizer').minimize(texture_loss,
                                                                                                var_list=texture_vars,
                                                                                                name='min_texture')
    ###========================== RESTORE MODEL =============================### 
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    tl.layers.initialize_global_variables(sess)

    init_restorer.restore(sess, checkpoint_dir + '/gan.ckpt')

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training

    train_hr_imgs = tl.prepro.threading_data(train_hr_imgs, fn=crop_sub_imgs_fn, size=target_size[3])
    sample_imgs_target = train_hr_imgs[1000: 1000 + batch_size]
    sample_imgs_start = tl.prepro.threading_data(train_hr_imgs[1000: 1000 + batch_size], fn=downsample_fn, size=target_size[0],
                                                 target_size=target_size[3])

    for img_idx in range(sample_imgs_start.shape[0]):
        tl.vis.save_image(sample_imgs_start[img_idx], save_dir_model + '/_train_sample_{}_{}.png'.format(target_size[0],img_idx))
        tl.vis.save_image(sample_imgs_target[img_idx], save_dir_model + '/_train_sample_{}_{}.png'.format(target_size[3],img_idx))

    print('done saving samples')

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    #for epoch in range(0, ginit_epoch[3] + 1):
    for epoch in range(0, 41):
        epoch_time = time.time()
        total_texture_loss, n_iter = 0, 0

        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_target = train_hr_imgs[idx: idx + batch_size]
            b_imgs_start = tl.prepro.threading_data(train_hr_imgs[idx: idx + batch_size], fn=downsample_fn,
                                                    size=target_size[0], target_size=target_size[3])
            ## update G
            original_train, diff_train, learned_train, errT, errG, _ = sess.run([train_output, diff, net_texture.outputs, texture_loss, gradient_loss, texture_optim], {t_image: b_imgs_start, t_target_image: b_imgs_target})
            #learned_train = (learned_train - np.amin(learned_train))/(np.amax(learned_train) - np.min(learned_train)) * 2 - 1
            print(np.amin(diff_train), np.amax(diff_train))
            print(np.amin(learned_train), np.amax(learned_train))
            print(np.amin(original_train), np.amax(original_train))
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
            epoch, ginit_epoch[3], n_iter, time.time() - step_time, errT))
            total_texture_loss += errT
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
        epoch, ginit_epoch[3], time.time() - epoch_time, total_texture_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 2 == 0):
            out_original, diff_out, learned_out = sess.run([net_g_test.outputs, diff, net_texture_test.outputs],
                                 {t_image: sample_imgs_start, t_target_image: sample_imgs_target})
            print("[*] save images")
            #learned_out = (learned_out - np.amin(learned_out))/(np.amax(learned_out) - np.amin(learned_out)) * 2 - 1
            out_improved = out_original + learned_out
            for img_idx in range(out_original.shape[0]):
                tl.vis.save_image(out_original[img_idx], save_dir_model + '/train_{}_{}_original.png'.format(epoch,img_idx))
                tl.vis.save_image(out_improved[img_idx], save_dir_model + '/train_{}_{}_improved.png'.format(epoch,img_idx))
                tl.vis.save_image(diff_out[img_idx], save_dir_model + '/train_{}_{}_true_diff.png'.format(epoch,img_idx))
                tl.vis.save_image(learned_out[img_idx], save_dir_model + '/train_{}_{}_learned_diff.png'.format(epoch,img_idx))


        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            saver.save(sess, checkpoint_dir + '/texture.ckpt')


def train_gradient(train_hr_imgs):
    ## create folders to save result images and trained model
    save_dir_model = os.path.join(args.dataset_path, "train_gradient_{}_{}".format(args.dataset,args.checkpoint))
    tl.files.exists_or_mkdir(save_dir_model)
    checkpoint_dir = os.path.join(args.dataset_path, "checkpoint_{}_{}".format(args.dataset,args.checkpoint))
    tl.files.exists_or_mkdir(checkpoint_dir)


    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, target_size[0], target_size[0], 3],
                             name='t_image_input_to_generator')
    t_target_image = tf.placeholder('float32', [batch_size, target_size[3], target_size[3], 3], name='t_target_image')

    if args.model == "MSSRGAN_texture":
        net_g, _ = MSSRGAN_texture_g_2(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=3)
        net_texture = MS_texture(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE)

    ## test inference
        net_g_test, _ = MSSRGAN_texture_g_2(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=3)
        net_texture_test = MS_texture(net_g_test.outputs, is_train=False, reuse=tf.AUTO_REUSE)

    elif args.model == "MSSRGAN_derivative":
        net_g, _ = MSSRGAN_texture_g_derivative(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=3)
        net_texture = MS_texture(net_g.outputs, is_train=True, reuse=tf.AUTO_REUSE)

    ## test inference
        net_g_test, _ = MSSRGAN_texture_g_derivative(t_image, is_train=False, reuse=tf.AUTO_REUSE, pg=3)
        net_texture_test = MS_texture(net_g_test.outputs, is_train=False, reuse=tf.AUTO_REUSE)
     
    ####========================== DEFINE TRAIN OPS ==========================###i
    diff = t_target_image - net_g.outputs
    diff_positive = tf.keras.activations.relu(diff, threshold=.02)
    diff_negative = tf.multiply(tf.keras.activations.relu(tf.multiply(diff, -1), threshold=.02), -1)

    diff = tf.add(diff_negative, diff_positive)
    #mse_mask = tf.add(tf.multiply(tf.cast(tf.equal(diff, 0.), tf.float32), 3), 1)
    #penalized_output = tf.multiply(mse_mask, net_texture.outputs)
    diff_grad = tf.image.image_gradients(tf.image.rgb_to_grayscale(diff))
    print(type(diff_grad))
    print(type(diff_grad[0]))
    print(type(diff_grad[1]))
    print(diff_grad[0].shape)
    print(diff_grad[1].shape)
    diff_grad = tf.stack([diff_grad[0], diff_grad[1]], axis = 3)
    output_grad = tf.image.image_gradients(tf.image.rgb_to_grayscale(net_texture.outputs))
    output_grad = tf.stack([output_grad[0], output_grad[1]], axis = 3)
    texture_loss = tf.losses.mean_squared_error(output_grad, diff_grad)
    sample_grad  = tf.image.image_gradients(tf.image.rgb_to_grayscale(net_texture_test.outputs))
    sample_grad = tf.stack([sample_grad[0], sample_grad[1]], axis = 3)

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    texture_vars = tl.layers.get_variables_with_name('texture', True, True)

    g_old_vars = []
    for i in range(1, 4):
        old_vars = [var for var in g_vars if 'pg{}'.format(i) in var.name]
        g_old_vars = g_old_vars + old_vars

    init_restorer = tf.train.Saver(g_old_vars)

    saver = tf.train.Saver(texture_vars)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    texture_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, name='texture_optimizer').minimize(texture_loss,
                                                                                                var_list=texture_vars,
                                                                                                name='min_texture')
    ###========================== RESTORE MODEL =============================### 
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    tl.layers.initialize_global_variables(sess)

    init_restorer.restore(sess, checkpoint_dir + '/gan.ckpt')

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training

    train_hr_imgs = tl.prepro.threading_data(train_hr_imgs, fn=crop_sub_imgs_fn, size=target_size[3])
    sample_imgs_target = train_hr_imgs[1000: 1000 + batch_size]
    sample_imgs_start = tl.prepro.threading_data(train_hr_imgs[1000: 1000 + batch_size], fn=downsample_fn, size=target_size[0],
                                                 target_size=target_size[3])

    for img_idx in range(sample_imgs_start.shape[0]):
        tl.vis.save_image(sample_imgs_start[img_idx], save_dir_model + '/_train_sample_{}_{}.png'.format(target_size[0],img_idx))
        tl.vis.save_image(sample_imgs_target[img_idx], save_dir_model + '/_train_sample_{}_{}.png'.format(target_size[3],img_idx))

    print('done saving samples')

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    #for epoch in range(0, ginit_epoch[3] + 1):
    for epoch in range(0, 41):
        epoch_time = time.time()
        total_texture_loss, n_iter = 0, 0

        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_target = train_hr_imgs[idx: idx + batch_size]
            b_imgs_start = tl.prepro.threading_data(train_hr_imgs[idx: idx + batch_size], fn=downsample_fn,
                                                    size=target_size[0], target_size=target_size[3])
            ## update G
            original_train, diff_train, errT, _ = sess.run([output_grad, diff_grad, texture_loss, texture_optim], {t_image: b_imgs_start, t_target_image: b_imgs_target})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
            epoch, ginit_epoch[3], n_iter, time.time() - step_time, errT))
            total_texture_loss += errT
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
        epoch, ginit_epoch[3], time.time() - epoch_time, total_texture_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 2 == 0):
            diff_out, learned_out = sess.run([sample_grad, diff_grad],
                                 {t_image: sample_imgs_start, t_target_image: sample_imgs_target})
            print("[*] save images")
            for img_idx in range(diff_out.shape[0]):
                tl.vis.save_image(diff_out[img_idx,:,:,0], save_dir_model + '/train_{}_{}_true_diff_grad_x.png'.format(epoch,img_idx))
                tl.vis.save_image(learned_out[img_idx,:,:,0], save_dir_model + '/train_{}_{}_learned_diff_grad_x.png'.format(epoch,img_idx))
                tl.vis.save_image(diff_out[img_idx,:,:,1], save_dir_model + '/train_{}_{}_true_diff_grad_y.png'.format(epoch,img_idx))
                tl.vis.save_image(learned_out[img_idx,:,:,1], save_dir_model + '/train_{}_{}_learned_diff_grad_y.png'.format(epoch,img_idx))

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            saver.save(sess, checkpoint_dir + '/gradient.ckpt')




if __name__ == '__main__':
    print(args.dataset_path)
    print(os.path.join(args.dataset_path, args.dataset, 'train_images_hr'))
    train_hr_img_list = sorted(tl.files.load_file_list(path=os.path.join(args.dataset_path, args.dataset, 'samples_5000/'), regx='.*.jpg', printable=False))
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=os.path.join(args.dataset_path, args.dataset, 'samples_5000/'), n_threads=32)
    
    if "MSSRGAN" in args.model:
        for i in range(1,4):
            if args.texture_only:
                break
            train(i, train_hr_imgs)
        #train_gradient(train_hr_imgs)
        train_texture(train_hr_imgs)
    else:
        train(3, train_hr_imgs)
