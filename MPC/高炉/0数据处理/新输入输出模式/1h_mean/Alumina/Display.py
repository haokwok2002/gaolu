import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import random
import Narx


def correlation_display(data):
    d = data.corr()
    plt.subplots(figsize=(12, 12))
    sns.heatmap(d,
                cbar=True,
                cmap="RdBu_r",
                annot=True,  # 注入数字
                square=True,  # 单元格为正方形
                fmt='.2f',  # 字符串格式代码
                annot_kws={'size': 10})  # 当annot为True时，ax.text的关键字参数，即注入数字的字体大小
    plt.show()


def rvflns_only_display(flag, start_point, display_num, train_time, test_time, entry, labels_size,
                        train_features, train_labels, test_features, test_labels, train_num, test_num,
                        scaler_labels, ae_input_hidden_w, ae_hidden_b, rvflns_w):
    if flag:
        # 绘制测试集
        ae_hidden = Narx.sigmoid(np.dot(test_features, ae_input_hidden_w) + ae_hidden_b)
        labels_hat = np.dot(ae_hidden, rvflns_w)
        if start_point is None:
            start_point = random.randint(0, test_num - display_num)
        display_time = test_time[start_point: start_point + display_num]
        display_data_norm = test_labels[start_point: start_point + display_num, :]
        display_data_hat_norm = labels_hat[start_point: start_point + display_num, :]
        display_data = scaler_labels.inverse_transform(display_data_norm)
        display_data_hat = scaler_labels.inverse_transform(display_data_hat_norm)
    elif flag == 0:
        # 绘制训练集
        ae_hidden = Narx.sigmoid(np.dot(train_features, ae_input_hidden_w) + ae_hidden_b)
        labels_hat = np.dot(ae_hidden, rvflns_w)
        if start_point is None:
            start_point = random.randint(0, train_num - display_num)
        display_time = train_time[start_point: start_point + display_num]
        display_data_norm = train_labels[start_point: start_point + display_num, :]
        display_data_hat_norm = labels_hat[start_point: start_point + display_num, :]
        display_data = scaler_labels.inverse_transform(display_data_norm)
        display_data_hat = scaler_labels.inverse_transform(display_data_hat_norm)
    else:
        return False

    image_draw(labels_size, entry, display_time, display_data, display_data_hat,
               scaler_labels.data_min_, scaler_labels.data_max_)

    return True


def rvflns_ae_display(flag, start_point, display_num, train_time, test_time, entry, features_size,
                      train_features, test_features, train_num, test_num,
                      scaler_features, ae_input_hidden_w, ae_hidden_b, ae_hidden_output_w):
    if flag:
        # 绘制测试集
        ae_hidden = Narx.sigmoid(np.dot(test_features, ae_input_hidden_w) + ae_hidden_b)
        features_hat = np.dot(ae_hidden, ae_hidden_output_w)
        if start_point is None:
            start_point = random.randint(0, test_num - display_num)
        display_time = test_time[start_point: start_point + display_num]
        display_data_norm = test_features[start_point: start_point + display_num, :]
        display_data_hat_norm = features_hat[start_point: start_point + display_num, :]
        display_data = scaler_features.inverse_transform(display_data_norm)
        display_data_hat = scaler_features.inverse_transform(display_data_hat_norm)
    elif flag == 0:
        # 绘制训练集
        ae_hidden = Narx.sigmoid(np.dot(train_features, ae_input_hidden_w) + ae_hidden_b)
        features_hat = np.dot(ae_hidden, ae_hidden_output_w)
        if start_point is None:
            start_point = random.randint(0, train_num - display_num)
        display_time = train_time[start_point: start_point + display_num]
        display_data_norm = train_features[start_point: start_point + display_num, :]
        display_data_hat_norm = features_hat[start_point: start_point + display_num, :]
        display_data = scaler_features.inverse_transform(display_data_norm)
        display_data_hat = scaler_features.inverse_transform(display_data_hat_norm)
    else:
        return False

    # 数据显示
    for i in range(features_size):
        plt.figure(i)
        plt.title("RVFLNs {}".format(entry[i]))
        plt.plot(display_time, display_data[:, i], label='Real', marker='o', markersize=3)
        plt.plot(display_time, display_data_hat[:, i], label='Predicted', marker='o', markersize=3)
        plt.ylim(scaler_features.data_min_[i] - 2, scaler_features.data_max_[i] + 2)
        plt.ylabel(entry[i])
        plt.xlabel('time')
        plt.legend()
        plt.show()


def rvflns_no_pca_display(flag, start_point, display_num, train_time, test_time, entry, labels_size,
                          train_features, train_labels, test_features, test_labels, train_num, test_num,
                          scaler_labels, rvflns_input_hidden_w, ae_hidden_b, rvflns_hidden_output_w):
    if flag:
        # 绘制测试集
        ae_hidden = Narx.sigmoid(np.dot(test_features, rvflns_input_hidden_w) + ae_hidden_b)
        labels_hat = np.dot(ae_hidden, rvflns_hidden_output_w)
        if start_point is None:
            start_point = random.randint(0, test_num - display_num)
        display_time = test_time[start_point: start_point + display_num]
        display_data_norm = test_labels[start_point: start_point + display_num, :]
        display_data_hat_norm = labels_hat[start_point: start_point + display_num, :]
        display_data = scaler_labels.inverse_transform(display_data_norm)
        display_data_hat = scaler_labels.inverse_transform(display_data_hat_norm)
    elif flag == 0:
        # 绘制训练集
        ae_hidden = Narx.sigmoid(np.dot(train_features, rvflns_input_hidden_w) + ae_hidden_b)
        labels_hat = np.dot(ae_hidden, rvflns_hidden_output_w)
        if start_point is None:
            start_point = random.randint(0, train_num - display_num)
        display_time = train_time[start_point: start_point + display_num]
        display_data_norm = train_labels[start_point: start_point + display_num, :]
        display_data_hat_norm = labels_hat[start_point: start_point + display_num, :]
        display_data = scaler_labels.inverse_transform(display_data_norm)
        display_data_hat = scaler_labels.inverse_transform(display_data_hat_norm)
    else:
        return False

    image_draw(labels_size, entry, display_time, display_data, display_data_hat,
               scaler_labels.data_min_, scaler_labels.data_max_)

    return True


def ae_pca_rvflns_results_display(flag, start_point, display_num, train_time, test_time, entry, labels_size,
                                  train_features, train_labels, test_features, test_labels, train_num, test_num,
                                  scaler_labels, rvflns_input_hidden_w, ae_hidden_b, transfer_matrix,
                                  rvflns_hidden_output_w):
    if flag:
        # 绘制测试集
        ae_hidden = Narx.sigmoid(np.dot(test_features, rvflns_input_hidden_w) + ae_hidden_b)
        reduced_ae_hidden = np.dot(ae_hidden, transfer_matrix)
        labels_hat = np.dot(reduced_ae_hidden, rvflns_hidden_output_w)
        if start_point is None:
            start_point = random.randint(0, test_num - display_num)
        display_time = test_time[start_point: start_point + display_num]
        display_data_norm = test_labels[start_point: start_point + display_num, :]
        display_data_hat_norm = labels_hat[start_point: start_point + display_num, :]
        display_data = scaler_labels.inverse_transform(display_data_norm)
        display_data_hat = scaler_labels.inverse_transform(display_data_hat_norm)
    elif flag == 0:
        # 绘制训练集
        ae_hidden = Narx.sigmoid(np.dot(train_features, rvflns_input_hidden_w) + ae_hidden_b)
        reduced_ae_hidden = np.dot(ae_hidden, transfer_matrix)
        labels_hat = np.dot(reduced_ae_hidden, rvflns_hidden_output_w)
        if start_point is None:
            start_point = random.randint(0, train_num - display_num)
        display_time = train_time[start_point: start_point + display_num]
        display_data_norm = train_labels[start_point: start_point + display_num, :]
        display_data_hat_norm = labels_hat[start_point: start_point + display_num, :]
        display_data = scaler_labels.inverse_transform(display_data_norm)
        display_data_hat = scaler_labels.inverse_transform(display_data_hat_norm)
    else:
        return False

    image_draw(labels_size, entry, display_time, display_data, display_data_hat,
               scaler_labels.data_min_, scaler_labels.data_max_)

    return True


def image_draw(labels_size, entry, display_time, display_data, display_data_hat, labels_min, labels_max):
    # 数据显示
    for i in range(labels_size):
        plt.figure(i)
        plt.title("RVFLNs {}".format(entry[-labels_size + i]))
        plt.plot(display_time, display_data[:, i], label='Real', marker='o', markersize=3)
        plt.plot(display_time, display_data_hat[:, i], label='Predicted', marker='o', markersize=3)
        plt.ylim(labels_min[i] - 2, labels_max[i] + 2)
        plt.ylabel(entry[-labels_size + i])
        plt.xlabel('time')
        plt.legend()
        plt.show()
