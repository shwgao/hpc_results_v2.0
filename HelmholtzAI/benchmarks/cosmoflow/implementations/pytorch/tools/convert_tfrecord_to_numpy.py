import tensorflow as tf
import numpy as np
import argparse
import os
from glob import glob


def parse_tfrecord(example_proto):
    # 解析特征，你需要根据你的.tfrecords文件的结构来调整
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        # 添加或修改其他特征
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def convert_to_numpy(tfrecord_files, output_dir, compression_type):
    for tfrecord_file in tfrecord_files:
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type=compression_type)
        parsed_dataset = raw_dataset.map(parse_tfrecord)

        # 假设每个.tfrecord文件只包含一个example
        for parsed_record in parsed_dataset:
            image_raw = parsed_record['image_raw'].numpy()
            image = np.frombuffer(image_raw, dtype=np.uint8)
            # 根据需要调整形状和类型

            # 构建输出文件路径
            base_name = os.path.basename(tfrecord_file)
            npy_filename = base_name.replace('.tfrecord', '.npy')
            output_path = os.path.join(output_dir, npy_filename)

            # 保存为.npy文件
            np.save(output_path, image)
            print(f'Saved {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='Input directory of tfrecord files')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory for npy files')
    parser.add_argument('-c', '--compression_type', type=str, default='', help='Compression type of tfrecord files')
    args = parser.parse_args()
    args.input_dir = '../data/cosmoUniverse_2019_05_4parE_tf_v2_mini/validation'
    args.output_dir = '../data/validation'

    # 创建输出目录，如果它不存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 获取所有.tfrecord文件
    tfrecord_files = glob(os.path.join(args.input_dir, '*.tfrecord'))

    # 转换文件
    convert_to_numpy(tfrecord_files, args.output_dir, args.compression_type)


if __name__ == '__main__':
    main()
