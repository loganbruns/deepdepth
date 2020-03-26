
""" Compress focal stack images and shard across tfrecord files """

import tensorflow as tf
import glob

def main():

    dataset = tf.data.TFRecordDataset(glob.glob('data/nyu_focal_stack*.tfrecord'))
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    PREFIX = 'data/compressed_nyu_focal_stack_'
    NUM_SHARDS = 96

    def reduce_func(key, dataset):
        filename = tf.strings.join([PREFIX, tf.strings.as_string(key), '.tfrecord'])
        writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
        writer.write(dataset.map(lambda _, x: x))
        return tf.data.Dataset.from_tensors(filename)
    
    dataset = dataset.enumerate()
    dataset = dataset.apply(tf.data.experimental.group_by_window(
        lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max
    ))

    for f in dataset:
        print(f'Wrote {f}.')

if __name__ == '__main__':
    main()
    
