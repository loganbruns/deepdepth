
""" Dataset Transforms """


import multiprocessing
import numpy as np

import tensorflow as tf


def random_crop_example(images, depth, focal_stack, height, width):
    """ Create a random crop into a pair of images in an example """

    y_size, x_size = tf.shape(images[0,:,:,:])[0], tf.shape(images[0,:,:,:])[1]
    y_offset = tf.random.uniform([1], 0, y_size - height, dtype=tf.int32)[0]
    x_offset = tf.random.uniform([1], 0, x_size - width, dtype=tf.int32)[0]

    images = tf.image.crop_to_bounding_box(images, y_offset, x_offset, height, width)
    depth = tf.image.crop_to_bounding_box(tf.expand_dims(depth, -1), y_offset, x_offset, height, width)
    depth = tf.squeeze(depth)

    return images, depth, focal_stack


def random_crop_dataset(dataset, height, width):
    """ Create a random crop into a pair of images in dataset """

    return dataset.map(
        lambda images, depth, focal_stack: random_crop_example(images, depth, focal_stack, height, width),
        num_parallel_calls=multiprocessing.cpu_count()
    )


def simulate_random_focal_lengths(image, depth, num_images, max_blur=4):
    quantized_depth, _, _ = tf.quantization.quantize(depth, 0, 10, tf.quint8)
    quantized_depth = tf.cast(quantized_depth, tf.int16)

    # Compute per depth masks
    mask_matrix = []
    for i in range(256):
        mask = tf.equal(quantized_depth, tf.cast(i, tf.int16) * tf.ones(depth.shape, tf.int16))
        mask_matrix.append(mask)
    mask_matrix = tf.stack(mask_matrix)
    mask_matrix = tf.cast(mask_matrix, tf.float32)
    mask_matrix = tf.expand_dims(mask_matrix, -1)

    focal_lengths = tf.cast(tf.cast(tf.random.uniform([num_images-1], 0, 255), tf.int16), tf.float32)
    images = [image]

    batch_image = tf.expand_dims(tf.transpose(image, [2, 0, 1]), -1)
    for j in range(num_images-1):
        focal_length = focal_lengths[j]

        # Compute blurred image matrix and 
        blurred_matrix = []
        for i in range(256):
            # Compute blurred images using gaussian kernel
            d = tf.abs(i - focal_length)
            sigma = (d / 256) * max_blur
            kernelSize = float(int(4.5 * sigma))
            dist = tf.compat.v1.distributions.Normal(0., sigma)
            vals = dist.prob(tf.range(start = -kernelSize, limit = kernelSize + 1, dtype = tf.float32))
            kernel = tf.einsum('i,j->ij', vals, vals)
            kernel = kernel / tf.reduce_sum(kernel)
            kernel = kernel[:, :, tf.newaxis, tf.newaxis]
            blurred_matrix.append(tf.transpose(tf.squeeze(tf.nn.conv2d(batch_image, kernel, strides=[1, 1, 1, 1], padding="SAME")), [1, 2, 0]))

        blurred_matrix = tf.stack(blurred_matrix)
        blurred_matrix = tf.cast(blurred_matrix, tf.float32)
        nonblurred_matrix = tf.tile(tf.expand_dims(image, 0), [256, 1, 1, 1])
        blurred_matrix = tf.where(tf.math.is_nan(blurred_matrix), nonblurred_matrix, blurred_matrix)

        blurred_image = tf.multiply(tf.cast(blurred_matrix, tf.float32), tf.cast(mask_matrix, tf.float32))
        blurred_image = tf.reduce_sum(blurred_image, 0)
        images.append(blurred_image)

    images = tf.stack(images)
    focal_lengths = [0] + focal_lengths
    focal_lengths = 10. * focal_lengths / 255.

    return images, depth, focal_lengths


def random_focal_stack(dataset, num_images):
    """ Create a focal stack and add it to a dataset """

    return dataset.map(
        lambda image, depth: simulate_random_focal_lengths(image, depth, num_images),
#        num_parallel_calls=multiprocessing.cpu_count()
    )
