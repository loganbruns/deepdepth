""" NYUv2 Depth Dataset Reader """

import tensorflow as tf

def NYUv2Dataset(filename, split=True):
    """ Create a dataset to read NYUv2 dataset """

    feature_description = {
        'image': tf.io.FixedLenFeature([3, 640, 480], tf.float32),
        'depth': tf.io.FixedLenFeature([640, 480], tf.float32),
        'shape': tf.io.FixedLenFeature([3], tf.int64),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = example['image']
        image /= 255
        image = tf.transpose(image, [2, 1, 0])
        depth = example['depth']
        depth = tf.transpose(depth, [1, 0])
        return (image, depth)

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)

    if split:
        train_size = 1200
        test_size = 125

        train = dataset.take(train_size).cache().shuffle(train_size)
        test = dataset.skip(train_size).take(test_size).cache().shuffle(test_size)
        val = dataset.skip(train_size+test_size).take(test_size).cache().shuffle(test_size)

        return (train, val, test)
    else:
        return dataset

def NYUv2FocalDataset(filename, split=True):
    """ Create a dataset to read NYUv2 dataset with precomputed focal stacks """

    feature_description = {
        'images': tf.io.FixedLenFeature([6, 480, 640, 3], tf.float32),
        'depth': tf.io.FixedLenFeature([480, 640], tf.float32),
        'focal_lengths': tf.io.FixedLenFeature([5], tf.float32),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return (
            example['images'],
            example['depth'],
            tf.concat(([0.], example['focal_lengths']), axis=0)
        )

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    if split:
        train_size = 614
        test_size = 5

        train = dataset.take(train_size).cache().shuffle(train_size)
        test = dataset.skip(train_size).take(test_size).cache().shuffle(test_size)
        val = dataset.skip(train_size+test_size).take(test_size).cache().shuffle(test_size)

        return (train, val, test)
    else:
        return dataset
