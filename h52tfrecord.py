
""" Convert NYUv2 HDF5 records into tf examples in tfrecord files """

import tensorflow as tf
import numpy as np

def main():

    images = tf.keras.utils.HDF5Matrix('data/nyu_depth_v2.h5', '/images')
    depths = tf.keras.utils.HDF5Matrix('data/nyu_depth_v2.h5', '/depths')

    dataset = tf.data.Dataset.from_tensor_slices((images, depths))

    def serialize_example(image, depth):
        image = np.array(image)
        depth = np.array(depth)
        
        feature = {
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=image.reshape(-1))),
            'depth': tf.train.Feature(float_list=tf.train.FloatList(value=depth.reshape(-1))),
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_serialize_example(image, depth):
        tf_string = tf.py_function(
            serialize_example,
            (image, depth),
            tf.string)
        return tf.reshape(tf_string, ())

    serialized_dataset = dataset.map(tf_serialize_example)

    writer = tf.data.experimental.TFRecordWriter('data/nyu_depth_v2.tfrecord')
    writer.write(serialized_dataset)

if __name__ == '__main__':
    main()
