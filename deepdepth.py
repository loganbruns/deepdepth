
""" Train Deep Depth Model """

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import argparse

import tensorflow as tf

import numpy as np

from depth_model import DepthModel
from nyu import NYUv2FocalDataset
from data_transforms import random_crop_dataset

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('experiment_name', None, 'Name of experiment to train and run.')

flags.DEFINE_string('gpu', '0', 'GPU to use')

flags.DEFINE_integer('batch_size', 32, 'Batch size')

flags.DEFINE_integer('context_length', 6, 'Context length- number of focal images ')

def depth_to_image(depth_map):
    depth_map = depth_map - tf.reduce_min(depth_map)
    depth_map = depth_map / tf.reduce_max(depth_map)
    # Apply gamma correction
    depth_map = tf.math.pow(depth_map, 1/2.2)
    depth_map = tf.stack([depth_map, depth_map, depth_map], -1)
    return depth_map

def main(unparsed_argv):
    """start main training loop"""

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # Set up experiment dir
    experiment_dir = f'./experiments/{FLAGS.experiment_name}'
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # Load model
    model = DepthModel(FLAGS.context_length)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, f'{experiment_dir}/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # Set up experimental logging
    train_log_dir = f'{experiment_dir}/logs/train'
    test_log_dir = f'{experiment_dir}/logs/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)        

    # Load NYUv2 depth dataset
    train, val, test = NYUv2FocalDataset('data/nyu_focal_stack.tfrecord')

    train = random_crop_dataset(train, 240, 320)
    val = random_crop_dataset(val, 240, 320)
    test = random_crop_dataset(test, 240, 320)

    train = train.repeat().batch(FLAGS.batch_size).prefetch(128)
    val = val.batch(FLAGS.batch_size)
    test = test.batch(FLAGS.batch_size)

    # Start training loop
    once_per_train = False
    starttime = time.time()
    startstep = int(ckpt.step)
        
    once_per_epoch = False
    for images, depth, focal_lengths in train:
        predictions = model.train_step(images, depth, focal_lengths)
        ckpt.step.assign_add(1)

        if not once_per_train:
            model.summary()
            once_per_train = True

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', model.train_loss.result(), step=int(ckpt.step))
            tf.summary.scalar('ssim', model.train_ssim.result(), step=int(ckpt.step))
            tf.summary.scalar('psnr', model.train_psnr.result(), step=int(ckpt.step))

            if int(ckpt.step) % 500 == 0:
                for i in range(FLAGS.context_length):
                    tf.summary.image(f'context_image_{i}', images[:,i,:,:], step=int(ckpt.step)-1, max_outputs=1)
                tf.summary.image('real_depth_map', depth_to_image(depth), step=int(ckpt.step)-1, max_outputs=1)
                tf.summary.image('pred_depth_map', depth_to_image(predictions), step=int(ckpt.step)-1, max_outputs=1)
                predictions = model.train_step(images, depth, focal_lengths, 1)
                tf.summary.image('single_depth_map', depth_to_image(predictions), step=int(ckpt.step)-1, max_outputs=1)
                
        if int(ckpt.step) % 100 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("Training loss {:1.2f}".format(model.train_loss.result()))

        if int(ckpt.step) % 500 == 0:
            with test_summary_writer.as_default():
                for test_images, test_depth, test_focal_lengths in val:
                    test_predictions = model.test_step(test_images, test_depth, test_focal_lengths)
                    for i in range(FLAGS.context_length):
                        tf.summary.image(f'context_image_{i}', test_images[:,i,:,:], step=int(ckpt.step), max_outputs=1)
                    tf.summary.image('real_depth_map', depth_to_image(test_depth), step=int(ckpt.step), max_outputs=1)
                    tf.summary.image('pred_depth_map', depth_to_image(test_predictions), step=int(ckpt.step), max_outputs=1)
                    test_predictions = model.test_step(test_images, test_depth, test_focal_lengths, 1)
                    tf.summary.image('single_depth_map', depth_to_image(test_predictions), step=int(ckpt.step), max_outputs=1)

                print(f"{int(ckpt.step)}: test loss={model.test_loss.result()}, ssim={model.test_ssim.result()}, psnr={model.test_psnr.result()}")
                tf.summary.scalar('loss', model.test_loss.result(), step=int(ckpt.step))
                tf.summary.scalar('ssim', model.test_ssim.result(), step=int(ckpt.step))
                tf.summary.scalar('psnr', model.test_psnr.result(), step=int(ckpt.step))

            template = 'Step {}, Loss: {}, SSIM: {}, PSNR: {}, Test Loss: {}, SSIM: {}, PSNR: {}, Sec/Iters: {}'
            print (template.format(int(ckpt.step),
                                   model.train_loss.result(),
                                   model.train_ssim.result(),
                                   model.train_psnr.result(),
                                   model.test_loss.result(),
                                   model.test_ssim.result(),
                                   model.test_psnr.result(),
                                   (time.time()-starttime)/(int(ckpt.step)-startstep)))
            starttime = time.time()
            startstep = int(ckpt.step)

            #tf.saved_model.save(model, f'{experiment_dir}/tf_model/{int(ckpt.step)}/')

if __name__ == '__main__':
    app.run(main)
