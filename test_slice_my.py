# the log is same as sliding_window_test_my

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import pydicom
import traceback
import time
slim = tf.contrib.slim
# from network import DNNs, DNNs_arg_scope
from xception4 import xception as DNNs
from xception4 import xception_arg_scope as DNNs_arg_scope
# from resnet import DNNs, DNNs_arg_scope
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import sklearn
from skimage import feature
import joblib


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parse_function(example_proto):
    features = tf.parse_single_example(example_proto,
                                       features={
                                           'imaString': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })
    image = tf.decode_raw(features['imaString'], tf.float32)
    image = tf.reshape(image, [size, size, samples_channel])
    label = features['label']
    return image, label


def copy_label_scan(z_list, spacing, effect, dcm_dir, save_dir):
    for filename in os.listdir(os.path.join(dcm_dir,'fake')):
        real_mat = pydicom.read_file(os.path.join(dcm_dir,'real',filename)).pixel_array
        fake_mat = pydicom.read_file(os.path.join(dcm_dir,'fake',filename)).pixel_array
        i = int(filename.split('.')[0])
        if (real_mat==fake_mat).all():
            new_filename = effect + '_%03d_' % i + 'T.dcm'
        else:
            new_filename = effect + '_%03d_' % i + 'F.dcm'
        shutil.copy(os.path.join(dcm_dir,'fake',filename),os.path.join(save_dir,new_filename))
    return


# to get how many samples in A Epoch
def getTheNumOfImgInAEpoch(tfrecord_dir):
    num_samples = 0
    tfrecords_to_count = [os.path.join(tfrecord_dir, file) for file in os.listdir(tfrecord_dir)]
    # tfrecords_to_count = tfrecords_to_count[:2]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    return num_samples


def generate_slide_window_coord():
    # 计算滑动窗口 求要裁剪的中心坐标 大致一个圆
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


def get_label(file_name):
    if file_name[-5] != 'F':
        label = 0
    else:
        label = 1
    return label

def cut_square(img_array, x, y, size):
    size = int(size / 2)
    # if x-size<0 or x+size>512 or y-size<0 or y+size>512:
    #     print('x,y=%d,%d cut image fail! size error!'%(x,y))
    #     return
    img = img_array[y - size:y + size, x - size:x + size]
    return img


def file_list_to_test_tfrecord(test_tfrecord_dir, data_dir, size):
    sw_coord = generate_slide_window_coord()
    img_number = 0
    writer = tf.python_io.TFRecordWriter(test_tfrecord_dir + os.sep + 'test_%.3d_.tfrecord' % img_number)
    neg_num = 0

    # def dcm_filename_weight(filename):
    #     return int(filename[:-4])
    # file_list = sorted(os.listdir(data_dir),key=dcm_filename_weight)
    file_list = sorted(os.listdir(data_dir))
    with open('./step_accuracy/org_test_filename_log.txt', 'w+') as file:
        for filename in file_list:
            try:
                label = get_label(filename)
                dcm = pydicom.read_file(data_dir + os.sep + filename)
                pix_array = dcm.pixel_array

                for x, y in sw_coord:
                    img = cut_square(pix_array, x, y, size)
                    img = np.reshape(img, [size, size, 1])
                    img = img.astype(np.float32)
                    imgString = img.tobytes()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'imaString': _bytes_features(imgString),
                        'label': _int64_features(label)
                    }))
                    writer.write(example.SerializeToString())

                img_number += 1
                print('filename:%s %d/%d samples have been processed.' % (filename, img_number, len(file_list)))
                file.write(filename + '\n')
            except Exception:
                traceback.print_exc()
                time.sleep(5)
                continue
        if writer is not None:
            writer.close()
    return


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


# def if_ct_true(negative_list):
    # for x0, y0 in negative_list:
    #     up = 0
    #     down = 0
    #     left = 0
    #     right = 0
    #     k = 5
    #     for i in range(1, k + 1):
    #         if [x0, y0 - i * stride] in negative_list:
    #             up += 1
    #         else:
    #             break
    #     for i in range(1, k + 1):
    #         if [x0, y0 + i * stride] in negative_list:
    #             down += 1
    #         else:
    #             break
    #     for i in range(1, k + 1):
    #         if [x0 - i * stride, y0] in negative_list:
    #             left += 1
    #         else:
    #             break
    #     for i in range(1, k + 1):
    #         if [x0 + i * stride, y0] in negative_list:
    #             right += 1
    #         else:
    #             break
    #     if (left + right) >= 4 and (up + down) >= 4:
    #         if ([x0 - stride, y0 - stride] in negative_list) and ([x0 - stride, y0 + stride] in negative_list) and (
    #                 [x0 + stride, y0 - stride] in negative_list) and ([x0 + stride, y0 + stride] in negative_list):
    #             return 1
def if_ct_true(heatmap,pca,svm):
    threshold = 100
    img = heatmap * (threshold - 1)
    img = np.uint8(img)
    img = feature.texture.greycomatrix(img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=threshold)[:, :, 0, :]

    x = pca.transform([img.flatten()])
    y = svm.predict(x)
    return y


def show_prediction_false(dcm_filename, windows_num, coord, predictions_list, labels_list, forge_probabilities_list):
    negative_list = []
    misjudge_list = []
    for i in range(windows_num):
        if predictions_list[i] == 1:
            negative_list.append(coord[i])
        if predictions_list[i] != labels_list[i]:
            misjudge_list.append(coord[i])
    # predict_label = if_ct_true(negative_list)

    # show dcm figure
    # ax1 = plt.subplot(1, 2, 1)
    # dcm = pydicom.read_file(data_dir + os.sep + dcm_filename)
    # ax1.imshow(dcm.pixel_array)
    # ax1.set_title('file_name:%s prediction:%d'%(dcm_filename,predict_label))
    # x_neg, y_neg = coord2xy(negative_list)
    # ax1.scatter(x_neg, y_neg, marker='x', color='red')
    # ax2 = plt.subplot(1, 2, 2)
    heatmap_data_mat = probabilities_list_to_heatmap_data(windows_num, coord, forge_probabilities_list)

    pca = joblib.load(os.path.join(ml_model_dir, 'pca.model'))
    svm = joblib.load(os.path.join(ml_model_dir, 'svm.model'))
    predict_label = if_ct_true(heatmap_data_mat,pca,svm)
    # sns.heatmap(heatmap_data_mat,vmin=0,vmax=1,ax=ax2,square=True)
    # sns.heatmap(heatmap_data_mat, ax=ax2, square=True)
    # plt.show()

    # save forge probabilities matrix as npy
    np.save('./npy_slice/' + dcm_filename[:-3] + 'npy', heatmap_data_mat)

    gt_label = get_label(dcm_filename)

    if gt_label == 0:
        if predict_label == 0:
            result_dict['TN'] += 1
        else:
            result_dict['FP'] += 1
    else:
        if predict_label == 0:
            result_dict['FN'] += 1
        else:
            result_dict['TP'] += 1
    precision = 0
    recall = 0
    accuracy = 0
    f1_score = 0
    if result_dict['TP'] > 0:
        precision = result_dict['TP'] / (result_dict['TP'] + result_dict['FP'])
        recall = result_dict['TP'] / (result_dict['TP'] + result_dict['FN'])
        accuracy = (result_dict['TP'] + result_dict['TN']) / \
                   (result_dict['TP'] + result_dict['TN'] + result_dict['FP'] + result_dict['FN'])
        f1_score = (2 * precision * recall) / (precision + recall)

    log_str = '\nfilename:{}  ground_true:{}  predict:{}\nTP:{} TN:{} FP:{} FN:{}' + \
              '\naccuracy:{}  precision:{}  recall:{} F1-score:{}\n'
    with open(total_ct_log_path, 'a+') as File:
        File.write(log_str.format(dcm_filename, gt_label, predict_label,
                                  result_dict['TP'], result_dict['TN'], result_dict['FP'], result_dict['FN'],
                                  accuracy, precision, recall, f1_score))
    return


if __name__ == '__main__':
    size = 32
    ct_size = 512
    stride = 4
    tfrecord_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/origin_tfrecord'
    data_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/luna_slice_data/malignant'
    unlabel_dcm_dir = r'/home/manager/hxx/PyWorkspace/CT-GAN-master/test_code'
    # copy_label_scan([54,56,76],2.5, 'remove', unlabel_dcm_dir, data_dir)
    file_list_to_test_tfrecord(tfrecord_dir, data_dir, size)  # first delete old tfrecord

    result_dict = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    # checkpointDir = r'./checkpoint'
    # checkpointDir = r'./ckp_bak'
    checkpointDir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/model_bak/Xception_SW/checkpoint'
    # checkpointDir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/model_bak/Resnet50_SW/checkpoint'
    # ml_model_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/ml_model'
    ml_model_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/model_bak/Xception_SW/ml_model/'
    # ml_model_dir = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/model_bak/Resnet50_SW/ml_model/'
    total_ct_log_path = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/step_accuracy/sw_test_total.txt'
    filename_log_path = r'/home/manager/hxx/PyWorkspace/GAN_small_forge_detection/step_accuracy/org_test_filename_log.txt'
    train_detail_str = '\n===== new log =====\n'

    num_class = 2
    samples_channel = 1
    bacthSize = 32
    keep_prob_set = 1.0
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    tf.logging.set_verbosity(tf.logging.INFO)

    fileList = [os.path.join(tfrecord_dir, file) for file in os.listdir(tfrecord_dir)]
    fileList.sort()

    # Get how many batches and samples in a epoch
    numSamplesPerEpoch = getTheNumOfImgInAEpoch(tfrecord_dir)
    numBatchesPerEpoch = int(np.ceil(numSamplesPerEpoch // bacthSize))

    # Read tfrecord file
    dataset = tf.data.TFRecordDataset(fileList)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.repeat(1)
    # dataset = dataset.shuffle(buffer_size=3000)
    dataset = dataset.batch(bacthSize)
    iterator = dataset.make_one_shot_iterator()
    nextImgs, nextLabels = iterator.get_next()


    def getImgesLabels(sess):
        imgs, labels = sess.run([nextImgs, nextLabels])
        return imgs, labels


    # IMGS = tf.placeholder(tf.float32, (bacthSize, threshold, threshold, 8))
    IMGS = tf.placeholder(tf.float32, (bacthSize, size, size, samples_channel))
    Labels = tf.placeholder(tf.int32, (bacthSize,))
    keep_prob = tf.placeholder(tf.float32)

    # Set default configuration
    with slim.arg_scope(DNNs_arg_scope()):  # 给DNN中的内容设置默认值，每个成员需要用@add_arg_scope修饰才行
        logits, end_points = DNNs(IMGS, num_classes=num_class, keep_prob=keep_prob, is_training=False)

    # Labels to one-hot encoding
    one_hot_labels = slim.one_hot_encoding(Labels, num_class)  # 将特征变为数字编码

    global_step = get_or_create_global_step()
    global_step_op = tf.assign(global_step, global_step + 1)

    # The probabilities of the samples in a batch
    probabilities = end_points['Predictions']
    predictions = tf.argmax(probabilities, 1)

    variables_to_restore = slim.get_variables_to_restore()  # 获取所有变量
    saver = tf.train.Saver(variables_to_restore)

    coord = generate_slide_window_coord()
    windows_num = len(coord)  # 一张ct窗口数

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
            imgs, labels = getImgesLabels(sess)

            probabilities_value, predictions_value = sess.run([probabilities, predictions],
                                                              feed_dict={IMGS: imgs, Labels: labels,
                                                                         keep_prob: keep_prob_set})
            predictions_value_list.extend(predictions_value)
            labels_list.extend(labels)
            forge_probabilities_list.extend(probabilities_value[:, 1])
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
                                      forge_probabilities_list[:windows_num])
                predictions_value_list = predictions_value_list[windows_num:]
                labels_list = labels_list[windows_num:]
                forge_probabilities_list = forge_probabilities_list[windows_num:]  #
                sample_index += 1

            globalStepCount = globalStepCount + 1
