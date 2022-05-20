import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import math
import cv2
import random
from skimage import feature

# from tensorflow.python.ops import control_flow_ops
slim = tf.contrib.slim
from network import DNNs, DNNs_arg_scope
# from xception4 import xception as DNNs
# from xception4 import xception_arg_scope as DNNs_arg_scope
# from resnet import DNNs, DNNs_arg_scope
import time
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python import pywrap_tensorflow
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
import pydicom


# Parse tfrecord file to rebuild images and labels
def _parse_function(example_proto):
    features = tf.parse_single_example(example_proto,
                                       features={
                                           'imaString': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })
    image = tf.decode_raw(features['imaString'], tf.float32)
    image = tf.reshape(image, [threshold, threshold, samples_channel])
    label = features['label']
    return image, label


# to get how many samples in A Epoch
def getTheNumOfImgInAEpoch(tfrecord_dir):
    num_samples = 0
    tfrecords_to_count = [os.path.join(tfrecord_dir, file) for file in os.listdir(tfrecord_dir)]
    # tfrecords_to_count = tfrecords_to_count[3:5]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    return num_samples


def coord2xy(coord_list):
    x_list = []
    y_list = []
    for x, y in coord_list:
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def probabilities_list_to_heatmap_data(windows_num, coord, forge_probabilities_list):
    heatmap_data_mat = np.zeros([121, 121])
    for i in range(windows_num):
        x, y = coord[i]
        x = x // 4 - 4
        y = y // 4 - 4
        # x:col, y:row
        heatmap_data_mat[y, x] = forge_probabilities_list[i]
    return heatmap_data_mat


def if_ct_true(negative_list):
    for x0, y0 in negative_list:
        up = 0
        down = 0
        left = 0
        right = 0
        k = 5
        for i in range(1, k + 1):
            if [x0, y0 - i * stride] in negative_list:
                up += 1
            else:
                break
        for i in range(1, k + 1):
            if [x0, y0 + i * stride] in negative_list:
                down += 1
            else:
                break
        for i in range(1, k + 1):
            if [x0 - i * stride, y0] in negative_list:
                left += 1
            else:
                break
        for i in range(1, k + 1):
            if [x0 + i * stride, y0] in negative_list:
                right += 1
            else:
                break
        if (left + right) >= 4 and (up + down) >= 4:
            if ([x0 - stride, y0 - stride] in negative_list) and ([x0 - stride, y0 + stride] in negative_list) and (
                    [x0 + stride, y0 - stride] in negative_list) and ([x0 + stride, y0 + stride] in negative_list):
                return 1
    return 0


def show_prediction_false(dcm_filename, windows_num, coord, predictions_list, labels_list, forge_probabilities_list):
    negative_list = []  # 被判断为假的点的索引
    misjudge_list = []  # 被误判的点的索引
    for i in range(windows_num):
        if predictions_list[i] == 1:
            negative_list.append(coord[i])
        if predictions_list[i] != labels_list[i]:
            misjudge_list.append(coord[i])

    # show dcm figure
    # dcm_data_dir = r'/home/manager/dataset/ComputerVision/Medical_Image/ctgan_my_generate/'
    # dcm_data_dir = r'/home/manager/hxx/PyWorkspace/GAN-MRI-master/CycleGAN/generate_images/synthetic_images/B/'
    # ax1 = plt.subplot(1,2,1)
    # dcm = pydicom.read_file(dcm_data_dir + dcm_filename)
    # ax1.imshow(dcm.pixel_array)
    # ax1.set_title(dcm_filename)
    # x_neg,y_neg = coord2xy(negative_list)
    # s_negative = ax1.scatter(x_neg, y_neg, marker='x', color='red')
    # x_list,y_list = coord2xy(misjudge_list)
    # s_misjudge = ax1.scatter(x_list, y_list, marker='+', color='blue')
    #
    # ax1.legend((s_negative, s_misjudge), ('negative', 'misjudge'), loc='best')
    # ax2 = plt.subplot(1,2,2)
    heatmap_data_mat = probabilities_list_to_heatmap_data(windows_num, coord, forge_probabilities_list)
    # # sns.heatmap(heatmap_data_mat,vmin=0,vmax=1,ax=ax2,square=True)
    # sns.heatmap(heatmap_data_mat, ax=ax2, square=True)
    # plt.savefig('./npy_data/' + dcm_filename[:-3] + 'pdf')
    # plt.show()

    # save forge probabilities matrix as npy
    np.save('./npy_data/' + dcm_filename[:-3] + 'npy', heatmap_data_mat)
    return


if __name__ == '__main__':
    result_dict = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    tfrecord_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/test_tfrecord'
    checkpointDir = r'./checkpoint'
    total_ct_log_path = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/step_accuracy/sw_test_total.txt'
    filename_log_path = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/step_accuracy/test_filename_log.txt'
    train_detail_str = '\n===== new log =====\n'

    num_class = 2
    samples_channel = 1
    bacthSize = 32
    threshold = 32
    learningRateDecayFactor = 0.85
    keep_prob_set = 1.0
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    # Set the verbosity to INFO level
    tf.logging.set_verbosity(tf.logging.INFO)

    fileList = [os.path.join(tfrecord_dir, file) for file in os.listdir(tfrecord_dir)]
    fileList.sort()

    # Read tfrecord file
    dataset = tf.data.TFRecordDataset(fileList)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    # dataset = dataset.shuffle(buffer_size=3000)
    dataset = dataset.batch(bacthSize)
    iterator = dataset.make_one_shot_iterator()
    nextImgs, nextLabels = iterator.get_next()

    IMGS = tf.placeholder(tf.float32, (bacthSize, threshold, threshold, samples_channel))
    Labels = tf.placeholder(tf.int32, (bacthSize,))
    keep_prob = tf.placeholder(tf.float32)

    # Set default configuration
    with slim.arg_scope(DNNs_arg_scope()):
        logits, end_points = DNNs(IMGS, num_classes=num_class, keep_prob=keep_prob, is_training=False)

    # Labels to one-hot encoding
    one_hot_labels = slim.one_hot_encoding(Labels, num_class)

    global_step = get_or_create_global_step()
    global_step_op = tf.assign(global_step, global_step + 1)

    # The probabilities of the samples in a batch
    probabilities = end_points['Predictions']
    predictions = tf.argmax(probabilities, 1)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    size = 32
    ct_size = 512
    stride = 4
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
    windows_num = len(coord)

    # Get how many batches and samples in a epoch
    numSamplesPerEpoch = getTheNumOfImgInAEpoch(tfrecord_dir)
    numBatchesPerEpoch = numSamplesPerEpoch // bacthSize
    # numSamplesPerEpoch = windows_num * 6000
    # numBatchesPerEpoch = math.ceil(numSamplesPerEpoch / bacthSize)

    lossAvg = 0
    lossTotal = 0
    with tf.Session(config=config) as sess:

        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())

        if os.listdir(checkpointDir):
            model_file = tf.train.latest_checkpoint(checkpointDir)
            saver.restore(sess, model_file)

            print('total samples: %d' % (bacthSize * numBatchesPerEpoch))
            print('the number of windows pre sample: %d' % windows_num)
            time.sleep(3)
        else:
            print('model not exist!')
            exit(1)

        globalStepCount = 0
        sample_index = 0
        predictions_value_list = []
        labels_list = []
        forge_probabilities_list = []  #
        with open(total_ct_log_path, 'w+') as File:
            File.write(train_detail_str)
            File.write('%d num_samples in a epoch' % numSamplesPerEpoch)
        for step in range(numBatchesPerEpoch):
            imgs, labels = sess.run([nextImgs, nextLabels])

            probabilities_value, predictions_value = sess.run([probabilities, predictions],
                                                              feed_dict={IMGS: imgs, Labels: labels,
                                                                         keep_prob: keep_prob_set})
            predictions_value_list.extend(predictions_value)
            labels_list.extend(labels)
            forge_probabilities_list.extend(probabilities_value[:, 1])  #
            if len(predictions_value_list) >= windows_num:
                with open(filename_log_path) as f:
                    for idx, line in enumerate(f):
                        if idx == sample_index:
                            dcm_filename = line.strip('\n')
                            print(idx, dcm_filename)
                            break
                show_prediction_false(dcm_filename, windows_num, coord,
                                      predictions_value_list[:windows_num],
                                      labels_list[:windows_num],
                                      forge_probabilities_list[:windows_num])  #
                predictions_value_list = predictions_value_list[windows_num:]
                labels_list = labels_list[windows_num:]
                forge_probabilities_list = forge_probabilities_list[windows_num:]  #
                sample_index += 1
            globalStepCount = globalStepCount + 1
