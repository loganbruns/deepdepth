
""" Dataset Transforms """

import multiprocessing

import tensorflow as tf

def random_crop_example(image, depth, height, width):
    """ Create a random crop into a pair of images in an example """

    y_size, x_size = tf.shape(image)[0], tf.shape(image)[1]
    y_offset = tf.random.uniform([1], 0, y_size - height, dtype=tf.int32)[0]
    x_offset = tf.random.uniform([1], 0, x_size - width, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image, y_offset, x_offset, height, width)
    depth = tf.image.crop_to_bounding_box(tf.expand_dims(depth, -1), y_offset, x_offset, height, width)
    depth = tf.squeeze(depth)

    return image, depth
    
def random_crop_dataset(dataset, height, width):
    """ Create a random crop into a pair of images in dataset """

    return dataset.map(
        lambda image, depth: random_crop_example(image, depth, height, width),
        num_parallel_calls=multiprocessing.cpu_count()
    )
