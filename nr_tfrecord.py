import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import tensorflow as tf
import numpy as np
import random
import cv2
from skimage import feature
import pydicom
import pandas as pd
import pywt
from skimage import filters
import copy
import time
import traceback
import copy


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_img_feature(img, size):
    # no features
    img = np.reshape(img, [size, size, 1])
    return img


def getLable(FileName):
    if FileName[0]=='n':
        return 1
    else:
        return 0


def cut_square(img_array, x, y, size):
    size = int(size / 2)
    img = img_array[y - size:y + size, x - size:x + size]
    return img


def get_example(img, label, size):
    img = get_img_feature(img, size)
    img = img.astype(np.float32)
    imgString = img.tobytes()
    # print(img.shape)

    example = tf.train.Example(features=tf.train.Features(feature={
        'imaString': _bytes_features(imgString),
        'label': _int64_features(label)
    }))
    return example


def generate_slide_window_coord():
    coord = []
    for y in range(size // 2, ct_size + 1 - size // 2, stride):
        h = abs(ct_size // 2 - y)
        w = int(np.sqrt((ct_size // 2) ** 2 - h ** 2))
        tmp = np.array(range(0, w + 1, stride))
        for x in np.concatenate((ct_size // 2 + tmp, ct_size // 2 - tmp)):
            if x - size // 2 < 0 or x + size // 2 > 512 or y - size // 2 < 0 or y + size // 2 > 512:
                continue
            else:
                coord.append([x, y])
    return coord


def file_list_to_test_tfrecord(test_tfrecord_dir, file_list, data_path, size):
    sw_coord = generate_slide_window_coord()
    img_number = 0
    writer = tf.python_io.TFRecordWriter(test_tfrecord_dir + os.sep + 'test_%.3d_.tfrecord' % img_number)
    with open('./step_accuracy/filename_log.txt', 'w+') as file:
        for filename in file_list:
            try:
                label = getLable(filename)
                dcm = pydicom.read_file(data_path + os.sep + filename)
                pix_array = dcm.pixel_array

                for x, y in sw_coord:
                    img = cut_square(pix_array, x, y, size)
                    example = get_example(img, label, size)
                    writer.write(example.SerializeToString())

                img_number += 1
                print('%d/%d samples have been processed.' % (img_number, len(file_list)))
                file.write(filename + '\n')

                if (img_number % 20) == 0:
                    writer.close()
                    writer = tf.python_io.TFRecordWriter(
                        test_tfrecord_dir + os.sep + 'test_%.3d_.tfrecord' % (img_number // 20))
            except Exception as e:
                file.write('break at %s' % filename)
                traceback.print_exc()
                time.sleep(5)
                continue

        if writer is not None:
            writer.close()
    return


def file_list_to_cv_tfrecord(tfrecord_path, file_list, data_path, size):
    img_number = 0
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    sw_coord = generate_slide_window_coord()
    for filename in file_list:
        try:
            label = getLable(filename)
            coord = random.sample(sw_coord, k=25)
            dcm = pydicom.read_file(data_path + os.sep + filename)
            pix_arr = dcm.pixel_array
            # for x,y in coord:
            for i in range(len(coord)):
                x, y = coord[i]
                img = cut_square(pix_arr, x, y, size)
                example = get_example(img, label, size)
                writer.write(example.SerializeToString())
                img_number += 1
                if img_number % 100 == 0:
                    print('%d/%d pictures have been processed' % (img_number, len(file_list) * len(coord)))
        except Exception as e:
            traceback.print_exc()
            continue
    writer.close()
    return


def file_list_to_train_tfrecord(train_tfrecord_dir, file_list, data_path, size):
    img_number = 0
    tfrecord_filename = 'train_%.3d_.tfrecord' % img_number
    writer = tf.python_io.TFRecordWriter(train_tfrecord_dir + os.sep + tfrecord_filename)
    sw_coord = generate_slide_window_coord()
    for filename in file_list:
        try:
            label = getLable(filename)
            coord = random.sample(sw_coord, k=25)
            dcm = pydicom.read_file(data_path + os.sep + filename)
            pix_arr = dcm.pixel_array
            # for x,y in coord:
            for i in range(len(coord)):
                x, y = coord[i]
                img = cut_square(pix_arr, x, y, size)
                example = get_example(img, label, size)
                writer.write(example.SerializeToString())
                img_number += 1
                if (img_number % 1000) == 0:
                    print('%d/%d pictures have been processed' % (img_number, len(file_list) * len(coord)))
                    writer.close()
                    tfrecord_filename = 'train_%.3d_.tfrecord' % (img_number // 1000)
                    writer = tf.python_io.TFRecordWriter(train_tfrecord_dir + os.sep + tfrecord_filename)
        except Exception as e:
            traceback.print_exc()
            continue

    if writer is not None:
        writer.close()
    return


def get_my_data_tfrecord(data_path, test_tfrecord_path, cv_tfrecord_path, train_tfrecord_path, threshold, size):
    file_list = []
    for file_name in os.listdir(data_path):
        if file_name[-1]=='m':
            file_list.append(file_name)
    random.shuffle(file_list)

    test_file_list = []
    cv_file_list = []
    train_file_list = []
    pos_num = 0
    neg_num = 0
    div1 = 1000
    div2 = 1500
    for file_name in file_list:
        if getLable(file_name) == 0:
            if pos_num >= div2:
                train_file_list.append(file_name)
            elif div1 <= pos_num < div2:
                cv_file_list.append(file_name)
            elif pos_num < div1:
                test_file_list.append(file_name)
            pos_num += 1
        else:
            if neg_num >= div2:
                train_file_list.append(file_name)
            elif div1 <= neg_num < div2:
                cv_file_list.append(file_name)
            elif neg_num < div1:
                test_file_list.append(file_name)
            neg_num += 1
    random.shuffle(test_file_list)
    random.shuffle(cv_file_list)
    random.shuffle(train_file_list)

    neg_num = 0
    for file_name in train_file_list:
        if getLable(file_name)==0:
            neg_num += 1
    print('neg_num:',neg_num)
    print(len(test_file_list),len(cv_file_list),len(train_file_list))
    time.sleep(5)

    print('===== test set tfrecord ======')
    file_list_to_test_tfrecord(test_tfrecord_path, test_file_list, data_path, size)

    print('===== cv set tfrecord ======')
    file_list_to_cv_tfrecord(cv_tfrecord_path + os.sep + 'cv_my_data.tfrecord', cv_file_list, data_path, size)

    print('===== train set tfrecord ======')
    file_list_to_train_tfrecord(train_tfrecord_path, train_file_list, data_path, size)
    return


if __name__ == '__main__':
    threshold = 192
    size = 32
    ct_size = 512
    stride = 4
    data_dir = r'/home/manager/hxx/PyWorkspace/GAN-MRI-master/CycleGAN/generate_images/synthetic_images/B'
    train_tfrecord_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/train_tfrecord'
    test_tfrecord_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/test_tfrecord'
    cv_tfrecord_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/cv_tfrecord'

    if not os.path.exists(train_tfrecord_dir):
        os.mkdir(train_tfrecord_dir)
    if not os.path.exists(test_tfrecord_dir):
        os.mkdir(test_tfrecord_dir)
    if not os.path.exists(cv_tfrecord_dir):
        os.mkdir(cv_tfrecord_dir)

    get_my_data_tfrecord(data_dir, test_tfrecord_dir, cv_tfrecord_dir, train_tfrecord_dir, threshold, size)
    print('Done!')
