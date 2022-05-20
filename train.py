import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
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

random.seed(1)

# if more than one tfrecord file you have generated, you can shuffle them to get higher randomness
def getNumEpochTfrecordWithShuffle(Path, Epoch):
    tfrecords = os.listdir(Path)
    for i in range(len(tfrecords)):
        tfrecords[i] = os.path.join(Path, tfrecords[i])
    shuffleNumEpochTfrecordsList = []
    for i in range(Epoch):
        random.shuffle(tfrecords)
        shuffleNumEpochTfrecordsList.append(tfrecords)
    return shuffleNumEpochTfrecordsList


# Parse tfrecord file to rebuild images and labels
def _parse_function(example_proto):
    features = tf.parse_single_example(example_proto,
                                       features={
                                           'imaString': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })
    image = tf.decode_raw(features['imaString'], tf.float32)
    # image = tf.reshape(image, [192, 192,8])
    image = tf.reshape(image, [threshold, threshold, samples_channel])
    label = features['label']
    return image, label


# to get how many samples in A Epoch
def getTheNumOfImgInAEpoch(tfrecordDir,trainFiles):
    num_samples = 0
    filePatternForCounting = trainFiles
    tfrecords_to_count = [os.path.join(tfrecordDir, file) for file in os.listdir(tfrecordDir) if
                          file.startswith(filePatternForCounting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    return num_samples


def cv_acc_of_early_stop(batch_size, num_samples, sess):
    num_batchs = num_samples // batch_size
    global_accuracy = 0
    accuracy_count = 0
    sess.run(cv_init_op)
    for step in range(num_batchs):
        imgs, labels = sess.run([next_cv_img, next_cv_label])
        cv_accuracy_update_value = sess.run(cv_accuracy,
                                            feed_dict={IMGS: imgs, Labels: labels, keep_prob: 1.0, is_train: False})
        accuracy_count += 1
        global_accuracy += cv_accuracy_update_value
    global_accuracy = global_accuracy / accuracy_count
    return global_accuracy


if __name__ == '__main__':
    train_tfrecord_dir = r'./train_tfrecord'
    cv_tfrecord_dir = r'./cv_tfrecord'
    trainFile = 'train'
    checkpointDir = r'./checkpoint'
    logDir = r'./step_accuracy/slide_window_train.txt'
    train_detail_str = '\n\n===== new train: keepprob={}, early_stop={}, feature Nan =====\n'

    num_class = 2
    early_stop_patience = 3
    samples_channel = 1
    bacthSize = 64
    threshold = 32
    howManyTimeShuffleFile = 5
    howManyRepeatFileList = 16
    initialLearningRate = 0.0005
    learningRateDecayFactor = 0.85
    keep_prob_set = 0.5
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    fileList = getNumEpochTfrecordWithShuffle(train_tfrecord_dir, howManyTimeShuffleFile)

    # Read tfrecord file
    dataset = tf.data.TFRecordDataset(fileList)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(howManyRepeatFileList)
    dataset = dataset.shuffle(buffer_size=3000)
    dataset = dataset.batch(bacthSize)
    iterator = dataset.make_one_shot_iterator()
    next_imgs, next_labels = iterator.get_next()

    cv_file_list = getNumEpochTfrecordWithShuffle(cv_tfrecord_dir,1)
    cv_dataset = tf.data.TFRecordDataset(cv_file_list)
    cv_dataset = cv_dataset.map(_parse_function)
    cv_dataset = cv_dataset.repeat(1)
    cv_dataset = cv_dataset.shuffle(buffer_size=3000)
    cv_dataset = cv_dataset.batch(bacthSize)
    cv_iterator = cv_dataset.make_one_shot_iterator()
    next_cv_img, next_cv_label = cv_iterator.get_next()
    cv_init_op = cv_iterator.make_initializer(dataset=cv_dataset)
    cv_num_samples = getTheNumOfImgInAEpoch(cv_tfrecord_dir,'cv')

    # Get how many batches and samples in a epoch
    numSamplesPerEpoch = getTheNumOfImgInAEpoch(train_tfrecord_dir,'train')
    numBatchesPerEpoch = numSamplesPerEpoch // bacthSize
    decay_steps = 600

    tf.logging.set_verbosity(tf.logging.INFO)

    IMGS = tf.placeholder(tf.float32, (bacthSize, threshold, threshold, samples_channel))
    Labels = tf.placeholder(tf.int32, (bacthSize,))
    keep_prob = tf.placeholder(tf.float32)

    # Set default configuration
    is_train = tf.placeholder(tf.bool)
    with slim.arg_scope(DNNs_arg_scope()):
        logits, end_points = DNNs(IMGS, num_classes=num_class, keep_prob=keep_prob, is_training=is_train)

    cv_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(tf.argmax(end_points['Predictions'], 1), tf.int32), Labels), tf.float32))

    # Labels to one-hot encoding
    one_hot_labels = slim.one_hot_encoding(Labels, num_class)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
    totalLoss = tf.losses.get_total_loss()
    globalStep = get_or_create_global_step()

    lr = tf.train.exponential_decay(
        learning_rate=initialLearningRate,
        global_step=globalStep,
        decay_steps=decay_steps,
        decay_rate=learningRateDecayFactor,
        staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Create the trainOp.
    trainOp = slim.learning.create_train_op(totalLoss, optimizer)
    predictions = tf.argmax(end_points['Predictions'], 1)

    # The probabilities of the samples in a batch
    probabilities = end_points['Predictions']
    accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, Labels)
    metrics_op = tf.group(accuracy_update, probabilities)


    def train_step(sess, trainOp, globalStep, imgs, labels):
        start_time = time.time()
        totalLoss, globalStepCount, _ = sess.run([trainOp, globalStep, metrics_op],
                                                 feed_dict={
                                                     IMGS: imgs,
                                                     Labels: labels,
                                                     keep_prob: keep_prob_set, is_train: True})
        time_elapsed = time.time() - start_time

        # Run the logging to print some results
        logging.info('global step %s: loss: %.4f (%.2f sec/step)', globalStepCount, totalLoss, time_elapsed)
        # return total loss and which step
        return totalLoss, globalStepCount


    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)

    saver = tf.train.Saver(max_to_keep=100)
    ckptPath = os.path.join(checkpointDir, 'mode.ckpt')
    lossAvg = 0
    lossTotal = 0
    # 用来early stop
    cv_acc = 0
    cv_acc_list = []
    best_cv_acc = 0
    patience = 0

    with open(logDir, 'w+') as File:
        File.write(train_detail_str.format(keep_prob_set, early_stop_patience))
    with tf.Session(config=config) as sess:

        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())

        if os.listdir(checkpointDir):
            model_file = tf.train.latest_checkpoint(checkpointDir)
            saver.restore(sess, model_file)
            print('total samples: %d' % (numBatchesPerEpoch * howManyTimeShuffleFile * howManyRepeatFileList))

        for step in range(numBatchesPerEpoch * howManyTimeShuffleFile * howManyRepeatFileList):
            imgs, labels = sess.run([next_imgs, next_labels])

            loss, globalStepCount = train_step(sess, trainOp, globalStep, imgs, labels)
            lossAvg += loss
            lossTotal += loss

            if step % 150 == 0:
                learning_rate_value, accuracy_value = sess.run([lr, accuracy], feed_dict={IMGS: imgs, Labels: labels})
                print('learning_rate_value: {}\n accuracy_value: {}'.format(learning_rate_value, accuracy_value))
                with open(logDir, 'a+') as File:
                    line_str = 'learining_rate: ' + str(learning_rate_value) + '  global_step: ' + str(
                        globalStepCount) + \
                               '  loss: ' + str(lossAvg / 150) + '  accuracy_value: ' + str(accuracy_value) + \
                               '  cv_acc: ' + str(cv_acc) + '\n'
                    print(line_str)
                    File.write(line_str)
                lossAvg = 0
                saver.save(sess, ckptPath, global_step=globalStep, write_meta_graph=False)
                if not os.path.exists(r'./checkpoint/*.meta'):
                    saver.export_meta_graph(r'./checkpoint/mode.ckpt.meta')

            # 提前停止
            if step % 3000 == 0:
                cv_acc = cv_acc_of_early_stop(bacthSize, cv_num_samples, sess)

                if cv_acc <= (best_cv_acc + 0.001):
                    patience += 1
                else:
                    best_cv_acc = cv_acc
                    patience = 0

                with open(logDir, 'a+') as File:
                    line_str = 'Epoch %d/%d' % (
                    step / numBatchesPerEpoch + 1, howManyTimeShuffleFile * howManyRepeatFileList) + \
                               '  global_step: ' + str(globalStepCount) + '  cv_acc: ' + str(cv_acc) + \
                               '  best_cv_acc: ' + str(best_cv_acc) + '\n'
                    print(line_str)
                    File.write(line_str)

                if patience > early_stop_patience:
                    print('early stop when %d step' % step)
                    break