import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

random.seed(1)


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def Scharr(img, threshold):
    x = cv2.Scharr(img, cv2.CV_16S, 1, 0)
    y = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    dst = cv2.addWeighted(abs(x), 0.5, abs(y), 0.5, 0)
    dst = np.clip(dst, 0, threshold - 1)
    return dst


def Sobel(img, threshold, Ksize=3):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=Ksize)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=Ksize)
    dst = cv2.addWeighted(abs(x), 0.5, abs(y), 0.5, 0)
    dst = np.clip(dst, 0, threshold - 1)
    return dst


def get_img_feature(img, size):
    if img.shape[0]!=size:
        print('invalid window size')
        return
    img = np.reshape(img, [size, size, 1]) # without features
    return img


def get_label(file_name):
    if file_name[-5] == 'T':
        return 0
    else:
        return 1


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


def get_coord_of_my_data(filename):
    if filename[7]=='3':
        return [0,0]
    df = pd.DataFrame(pd.read_csv(csv_path, header=0))
    for i, row in df.iterrows():
        if row.filename[:11] == (filename[:11]):
            if row.x<18 or row.x>493 or row.y<18 or row.y>493:
                print(filename, '!!!coord invaild!!!')
                return
            return [int(row.x), int(row.y)]
    print(filename, '!!!no log!!!')
    return


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


def file_list_to_total_image_tfrecord(effect, tfrecord_dir, file_list, data_path, size):
    sw_coord = generate_slide_window_coord()
    img_number = 0
    writer = tf.python_io.TFRecordWriter(tfrecord_dir + os.sep + effect + '_%.3d_.tfrecord' % img_number)
    neg_num = 0
    with open('./step_accuracy/' + effect + '_filename_log.txt', 'w+') as file:
        for filename in file_list:
            try:
                filename_label = get_label(filename)
                x_forge, y_forge = get_coord_of_my_data(filename)
                if filename_label == 1:  # 0真1假
                    if x_forge < 0 or y_forge < 0:
                        continue
                    neg_num += 1
                dcm = pydicom.read_file(data_path + os.sep + filename)
                pix_array = dcm.pixel_array

                for x, y in sw_coord:
                    label = 0
                    if filename_label == 1:
                        if abs(x - x_forge) < size and abs(y - y_forge) < size:
                            label = 1
                    img = cut_square(pix_array, x, y, size)
                    example = get_example(img, label, size)
                    writer.write(example.SerializeToString())

                img_number += 1
                print('%d/%d samples have been processed.' % (img_number, len(file_list)))
                file.write(filename + '\n')

                if (img_number % 20) == 0:
                    writer.close()
                    writer = tf.python_io.TFRecordWriter(
                        tfrecord_dir + os.sep + effect + '_%.3d_.tfrecord' % (img_number // 20))
            except Exception as e:
                # file.write('break at %s' % filename)
                traceback.print_exc()
                time.sleep(5)
                continue

        file.write('negative:%d' % neg_num)
        if writer is not None:
            writer.close()
    return


def cut_coord_list_form_file(label, filename):
    k = 25
    n = 10

    coord = []
    x0, y0 = get_coord_of_my_data(filename)
    if x0 < 0 or y0 < 0:
        return
    x_tmp = 1 if x0 < (ct_size // 2) else -1
    y_tmp = 1 if y0 < (ct_size // 2) else -1
    sw_coord = generate_slide_window_coord()
    if label == 0:  # and random.choice(range(5))!=0: # label 0真 1假
        coord_num = 0
        if x0 == 0 and y0 == 0:
            coord.extend(random.sample(sw_coord, k=k))
        else:
            for i in range(n):
                coord.append([x0 + x_tmp * (i % 5 - 2), y0 + y_tmp * (i // 5 - 2)])
                coord_num += 1
            coord.extend(random.sample(sw_coord, k=(k - coord_num)))
    else:
        for i in range(k):
            coord.append([x0 + x_tmp * (i % 7 - 2), y0 + y_tmp * (i // 7 - 2)])

    if len(coord) != k:
        print(filename + '!!! coord error')
    return coord


def file_list_to_every_window_tfrecord(effect, tfrecord_dir, file_list, data_path, size):
    img_number = 0
    tfrecord_filename = effect + '_%.3d_.tfrecord' % img_number
    writer = tf.python_io.TFRecordWriter(tfrecord_dir + os.sep + tfrecord_filename)
    for filename in file_list:
        try:
            label = get_label(filename)
            coord = cut_coord_list_form_file(label, filename)
            dcm = pydicom.read_file(data_path + os.sep + filename)
            pix_arr = dcm.pixel_array
            for i in range(len(coord)):
                x, y = coord[i]
                img = cut_square(pix_arr, x, y, size)
                example = get_example(img, label, size)
                writer.write(example.SerializeToString())
                img_number += 1
                if (img_number % 1000) == 0:
                    print('%d/%d pictures have been processed' % (img_number, len(file_list) * len(coord)))
                    writer.close()
                    tfrecord_filename = effect + '_%.3d_.tfrecord' % (img_number // 1000)
                    writer = tf.python_io.TFRecordWriter(tfrecord_dir + os.sep + tfrecord_filename)
        except Exception as e:
            traceback.print_exc()
            continue

    if writer is not None:
        writer.close()
    return


def get_inj_or_rem_filename(file_list, pattern):
    pos_list = []
    neg_list = []
    for file_name in file_list:
        if file_name[0] == pattern and file_name[15] == 'F':
            pos_list.append(file_name)
        if file_name[0] != pattern and file_name[15] == 'T':
            neg_list.append(file_name)
    min_num = min(len(pos_list),len(neg_list))
    pos_list = pos_list[:min_num]
    neg_list = neg_list[:min_num]
    new_file_list = pos_list+neg_list
    return new_file_list


def remove_extra_true(file_list):
    new_file_list = []
    for file_name in file_list:
        if file_name[7] != '3':
            new_file_list.append(file_name)
    return new_file_list


def replace_true_samples(file_list, data_dir, replace_num):
    new_file_list = copy.deepcopy(file_list)
    replace_idx = 0
    for i in range(len(file_list)):
        file_name = file_list[i]
        if file_name[7] != '3' and file_name[-5] == 'T':
            # print(file_name)
            new_file_list.remove(file_name)
            replace_idx += 1
            if replace_idx >= replace_num:
                # print('replace>5000')
                break
        else:
            continue
    for file_name in os.listdir(data_dir):
        if file_name[7] == '3':
            new_file_list.append(file_name)
    random.shuffle(new_file_list)
    return new_file_list


def get_my_data_tfrecord(data_dir, test_tfrecord_dir, cv_tfrecord_dir, train_tfrecord_dir, size):
    file_list = os.listdir(data_dir)
    file_list = remove_extra_true(file_list)
    # file_list = get_inj_or_rem_filename(file_list, 'i')
    random.shuffle(file_list)
    file_list = replace_true_samples(file_list, data_dir, 8850)

    test_file_list = []
    cv_file_list = []
    train_file_list = []
    pos_num = 0
    neg_num = 0
    div1 = 3000
    div2 = 4000
    for file_name in file_list:
        if get_label(file_name) == 0:
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
    print(len(test_file_list), len(cv_file_list), len(train_file_list),pos_num,neg_num)
    time.sleep(5)
    random.shuffle(test_file_list)
    random.shuffle(cv_file_list)
    random.shuffle(train_file_list)

    print('===== test tfrecord ======')
    file_list_to_total_image_tfrecord('test', test_tfrecord_dir, test_file_list, data_dir, size)

    print('===== cv tfrecord ======')
    file_list_to_every_window_tfrecord('cv', cv_tfrecord_dir, cv_file_list, data_dir, size)

    print('===== sliding window train tfrecord ======')
    file_list_to_every_window_tfrecord('train', train_tfrecord_dir, train_file_list, data_dir, size)

    return


if __name__ == '__main__':
    size = 32
    ct_size = 512
    stride = 4
    my_data_dir = r'/home/manager/dataset/ComputerVision/Medical_Image/ctgan_my_generate'
    csv_path = r'/home/manager/hxx/PyWorkspace/CT-GAN-master/data/test_vox.csv'
    train_tfrecord_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/train_tfrecord'
    test_tfrecord_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/test_tfrecord'
    cv_tfrecord_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/cv_tfrecord'

    get_my_data_tfrecord(my_data_dir, test_tfrecord_dir, cv_tfrecord_dir, train_tfrecord_dir, size)
    print('Done of the tfrecord of my data!')
    print('Done!')
