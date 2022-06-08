import os
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from skimage import feature
import joblib

random.seed(1)


def get_label(file_name):
    if file_name[-5] == 'T':
        return 0
    else:
        return 1
    # if file_name[0] == 'n':
    #     return 1
    # else:
    #     return 0


def get_result(gt_list, predict_list):
    result_dict = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for i in range(len(gt_list)):
        if gt_list[i] == 0:
            if predict_list[i] == 0:
                result_dict['TN'] += 1
            else:
                result_dict['FP'] += 1
        else:
            if predict_list[i] == 0:
                result_dict['FN'] += 1
            else:
                result_dict['TP'] += 1
    precision = result_dict['TP'] / (result_dict['TP'] + result_dict['FP'])
    recall = result_dict['TP'] / (result_dict['TP'] + result_dict['FN'])
    accuracy = (result_dict['TP'] + result_dict['TN']) / (
            result_dict['TP'] + result_dict['TN'] + result_dict['FP'] + result_dict['FN'])
    f1_score = (2 * precision * recall) / (precision + recall)
    print(precision, recall, accuracy, f1_score)
    # return precision,recall,accuracy,f1_score


def get_train_test(file_list, train_size):
    train_size = train_size // 2
    pos_num = 0
    neg_num = 0
    train_list = []
    test_list = []
    for file_name in file_list:
        if get_label(file_name) == 0:
            if pos_num < train_size:
                train_list.append(file_name)
            elif pos_num >= train_size:
                test_list.append(file_name)
            pos_num += 1
        else:
            if neg_num < train_size:
                train_list.append(file_name)
            elif neg_num >= train_size:
                test_list.append(file_name)
            neg_num += 1
    return train_list, test_list


def new_parameters_list(value):
    value_str = '%.10f' % value
    # print(value_str)
    if '1' in value_str:
        return [value * 0.6, value * 0.8, value, value * 2, value * 4]
    if '5' in value_str:
        temp = value / 5
        return [temp * 2, temp * 4, value, temp * 7, temp * 9]

def pca_keep_ratio(variance_list):
    v_all = 0
    for v in variance_list:
        v_all = v_all + v
    return v_all

if __name__ == '__main__':
    data_dir = r'./npy_data/'
    model_dir = r'./ml_model'
    total_ct_log_path = r'./step_accuracy/sw_test_total.txt'

    start_time = time.time()
    print('loading data...')
    file_list = os.listdir(data_dir)
    random.shuffle(file_list)
    gt_list = []
    data_list = []
    for file_name in file_list:
        gt_list.append(get_label(file_name))
        img = np.load(data_dir + file_name)

        # GLCM
        threshold = 100
        img = img * (threshold - 1)
        img = np.uint8(img)
        img = feature.texture.greycomatrix(img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],levels=threshold)[:, :, 0, :]

        # print(img.shape)
        data_list.append(img.flatten())

    # normalization
    # scaler = preprocessing.StandardScaler()
    # data_list = scaler.fit_transform(data_list)

    end_time = time.time()
    print('data have load. cost %.3f s' % (end_time - start_time))

    random_state = 0
    train_x, test_x, train_y, test_y = train_test_split(data_list, gt_list, test_size=0.2, stratify=gt_list, random_state=random_state)

    # pca
    start_time = end_time
    print('training PCA...')
    pca = PCA(n_components=256, svd_solver='randomized', whiten=True, random_state=random_state)
    pca.fit(train_x)
    train_x_pca = pca.transform(train_x)
    test_x_pca = pca.transform(test_x)
    end_time = time.time()
    print('PCA done. cost %.3f s' % (end_time - start_time))
    keep = pca_keep_ratio(pca.explained_variance_ratio_)
    pca_log = '\nPCA info: n_components='+str(pca.n_components_)+'; keep_ratio='+str(keep)
    print(pca_log)

    # svm
    start_time = end_time
    print('training SVM...')

    param_grid = {
        'C': [1e2, 5e2, 1e3, 5e3, 1e4],
        'gamma': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
    }

    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

    clf = clf.fit(train_x_pca, train_y)
    print("Best estimator found by grid search:")
    print('epoch 1:', clf.best_params_)

    # svm epoch2
    param_grid['C'] = new_parameters_list(clf.best_params_['C'])
    param_grid['gamma'] = new_parameters_list(clf.best_params_['gamma'])
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(train_x_pca, train_y)
    print('epoch 2:', clf.best_params_)

    y_predict = clf.predict(test_x_pca)
    end_time = time.time()
    print('SVM training done! cost %.3f s' % (end_time - start_time))

    print(classification_report(test_y, y_predict, digits=3))
    with open(total_ct_log_path, 'a+') as file:
        file.write(pca_log)
        file.write('\nBest estimator found by grid search:\n')
        file.write(str(clf.best_params_))
        file.write('\nClassification report:\n')
        file.write(str(classification_report(test_y, y_predict, digits=4)))

    # save models
    joblib.dump(pca,os.path.join(model_dir,'pca.model'))
    joblib.dump(clf,os.path.join(model_dir,'svm.model'))