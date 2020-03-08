
""" Train Deep Depth Model """

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import argparse

import tensorflow as tf

import numpy as np

from depth_model import DepthModel
from nyu import NYUv2Dataset
from data_transforms import random_crop_dataset

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('experiment_name', None, 'Name of experiment to train and run.')

flags.DEFINE_string('gpu', '0', 'GPU to use')

flags.DEFINE_integer('batch_size', 32, 'Batch size')

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
    model = DepthModel()
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
    train, val, test = NYUv2Dataset('data/nyu_depth_v2.tfrecord')

    train = random_crop_dataset(train, 360, 480)
    val = random_crop_dataset(val, 360, 480)
    test = random_crop_dataset(test, 360, 480)

    train = train.batch(FLAGS.batch_size)
    val = val.batch(FLAGS.batch_size)
    test = test.batch(FLAGS.batch_size)

    # Start training loop
    EPOCHS = 1000
    once_per_train = False
    for epoch in range(EPOCHS):
        starttime = time.time()
        startstep = int(ckpt.step)
        for image, depth in train:
            model.train_step(image, depth)
            ckpt.step.assign_add(1)

            if not once_per_train:
                model.summary()
                once_per_train = True

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', model.train_loss.result(), step=int(ckpt.step))
                
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("Training loss {:1.2f}".format(model.train_loss.result()))

        with test_summary_writer.as_default():
            for test_image, test_label in val:
                test_predictions = model.test_step(test_image, test_label)
                tf.summary.image('context_images', test_image, step=int(ckpt.step))
                tf.summary.image('real_depth_map', depth_to_image(test_label), step=int(ckpt.step))
                tf.summary.image('pred_depth_map', depth_to_image(test_predictions), step=int(ckpt.step))

            print(f"{int(ckpt.step)}: test loss={model.test_loss.result()}")
            tf.summary.scalar('loss', model.test_loss.result(), step=int(ckpt.step))

        template = 'Epoch {}, Loss: {}, Test Loss: {}, Sec/Iters: {}'
        print (template.format(epoch+1,
                               model.train_loss.result(),
                               model.test_loss.result(),
                               (time.time()-starttime)/(int(ckpt.step)-startstep)))
        #tf.saved_model.save(model, f'{experiment_dir}/tf_model/{int(ckpt.step)}/')

if __name__ == '__main__':
    app.run(main)
