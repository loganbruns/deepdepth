""" NYUv2 Depth Dataset Reader """

import tensorflow as tf

def NYUv2Dataset(filename):
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

    dataset = tf.data.TFRecordDataset('data/nyu_depth_v2.tfrecord')
    dataset = dataset.map(_parse_function)
    
    train_size = 1200
    test_size = 125

    train = dataset.take(train_size).cache().shuffle(2500)
    test = dataset.skip(train_size).take(test_size).cache()
    val = dataset.skip(train_size+test_size).cache()

    return (train, val, test)
