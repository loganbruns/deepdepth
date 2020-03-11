
""" Add focal stacks to NYUv2 tfrecord files """

import tensorflow as tf

from nyu import NYUv2Dataset
from data_transforms import random_focal_stack

def main():

    dataset = NYUv2Dataset('data/nyu_depth_v2.tfrecord', split=False)

    dataset = random_focal_stack(dataset, 6)

    def serialize_example(images, depth, focal_lengths):
        feature = {
            'images': tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(images, [-1]))),
            'depth': tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(depth, [-1]))),
            'focal_lengths': tf.train.Feature(float_list=tf.train.FloatList(value=focal_lengths)),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_serialize_example(images, depth, focal_lengths):
        tf_string = tf.py_function(
            serialize_example,
            (images, depth, focal_lengths),
            tf.string)
        return tf.reshape(tf_string, ())

    serialized_dataset = dataset.map(tf_serialize_example)

    writer = tf.data.experimental.TFRecordWriter('data/nyu_focal_stack.tfrecord')
    writer.write(serialized_dataset)

if __name__ == '__main__':
    main()
    
    
