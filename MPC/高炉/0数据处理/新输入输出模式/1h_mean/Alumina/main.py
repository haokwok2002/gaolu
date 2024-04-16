# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import Narx
import Display


def main():
    # 参数设置
    time_lag_order = 1  # 时滞阶数
    cor_display_flag = 0  # 相关图绘制标志
    dataset_shuffle_flag = 0  # 数据集打乱标志，0 不打乱，1打乱
    ae_hidden_size = 128  # 隐含层维数
    lam_l2 = 0.28  # L2 正则化系数
    pca_accu_var = 0.95  # pca 降维累计贡献率
    display_flag = 0  # 绘图参数，0 显示训练集，1 显示测试集
    display_start_point = 10  # 绘图时间起始，None 随机起始
    display_num = 400  # 绘图数据数量

    # 导入数据
    df = pd.read_excel("./Datasets/Alumina_dataset.xlsx", header=1)
    # 时间索取
    time = np.array(pd.to_datetime(df['Time']))[time_lag_order:].astype('datetime64[s]')
    # 各变量名称索取
    entry = pd.read_csv("./Datasets/entry.txt", header=None)
    entry = np.array(entry).astype(str)
    # 热力学图显示
    if cor_display_flag:
        Display.correlation_display(df.iloc[:, 2:])

    # 数据划分
    data = df.to_numpy()  # 类型转换
    col_split_point = 12  # 特征标签划分点
    original_features = data[:, 2:col_split_point].astype('float32')  # 特征
    original_labels = data[:, col_split_point:].astype('float32')  # 标签

    # 获取 NARX 特征
    features = original_features[time_lag_order:, :]
    for i in range(1, time_lag_order + 1):
        features = np.concatenate((features, original_features[time_lag_order - i: -i, :]), axis=1)
    for i in range(1, time_lag_order + 1):
        features = np.concatenate((features, original_labels[time_lag_order - i: -i, :]), axis=1)
    # 获取 NARX 标签
    labels = original_labels[time_lag_order:, :]

    # 特征、标签数量提取
    features_size = np.size(features, 1)  # 输入数据的特征数量
    labels_size = np.size(labels, 1)  # 输出数据的标签数量

    # 数据是否打乱
    if dataset_shuffle_flag:
        np.random.shuffle(features)
        np.random.shuffle(labels)

    # 数据归一化
    scaler_features = MinMaxScaler()
    scaler_labels = MinMaxScaler()
    features_norm = scaler_features.fit_transform(features)
    labels_norm = scaler_labels.fit_transform(labels)

    # 划分训练集和测试集
    split_point = [0.8, 0.2]
    train_features = features_norm[:int(len(features_norm) * split_point[0]), :]
    train_labels = labels_norm[:int(len(labels_norm) * split_point[0]), :]
    test_features = features_norm[int(len(features_norm) * split_point[0]):, :]
    test_labels = labels_norm[int(len(labels_norm) * split_point[0]):, :]
    train_num = len(train_features)
    test_num = len(test_features)
    train_time = time[:int(len(time) * split_point[0])]
    test_time = time[int(len(time) * split_point[0]):]

    # AE 自编码器网络构建
    # 输入权重和隐含层偏置随机产生
    ae_input_hidden_w = np.random.uniform(-1, 1, (features_size, ae_hidden_size))
    ae_hidden_b = np.random.uniform(-1, 1, (1, ae_hidden_size))
    # 输入数据映射至隐含层
    train_ae_hidden = Narx.sigmoid(np.dot(train_features, ae_input_hidden_w) + ae_hidden_b)

    if input("Proceed to the RVFLNs only?[y/n]") == 'y':
        # 仅 RVFLNs 直接求解输出
        # rvflns_w = Narx.rvflns_lsm_calculate(train_ae_hidden, train_labels)
        rvflns_w = Narx.rvflns_lsm_l2_calculate(train_ae_hidden, train_labels, lam_l2)
        Display.rvflns_only_display(display_flag, display_start_point, display_num, train_time, test_time, entry,
                                    labels_size,
                                    train_features, train_labels, test_features, test_labels, train_num, test_num,
                                    scaler_labels, ae_input_hidden_w, ae_hidden_b, rvflns_w)

    # 求解 AE 输出权值矩阵
    # ae_hidden_output_w = Narx.rvflns_lsm_calculate(train_ae_hidden, train_features)
    ae_hidden_output_w = Narx.rvflns_lsm_l2_calculate(train_ae_hidden, train_features, lam_l2)
    if input("Proceed to the AE display?[y/n]") == 'y':
        Display.rvflns_ae_display(display_flag, display_start_point, display_num, train_time, test_time, entry,
                                  features_size, train_features, test_features, train_num, test_num,
                                  scaler_features, ae_input_hidden_w, ae_hidden_b, ae_hidden_output_w)

    # 求解 RVFLNs 输入权值矩阵
    rvflns_input_hidden_w = np.transpose(ae_hidden_output_w)
    # 使用输出权重矩阵重新计算隐含层
    train_ae_hidden = Narx.sigmoid(np.dot(train_features, rvflns_input_hidden_w) + ae_hidden_b)
    if input("Proceed to the no PCA display?[y/n]") == 'y':
        # rvflns_hidden_output_w = Narx.rvflns_lsm_calculate(train_ae_hidden, train_labels)
        rvflns_hidden_output_w = Narx.rvflns_lsm_l2_calculate(train_ae_hidden, train_labels, lam_l2)
        Display.rvflns_no_pca_display(display_flag, display_start_point, display_num, train_time, test_time, entry,
                                      labels_size,
                                      train_features, train_labels, test_features, test_labels, train_num, test_num,
                                      scaler_labels, rvflns_input_hidden_w, ae_hidden_b, rvflns_hidden_output_w)

    # 隐含层 PCA 降维
    transfer_matrix, reduced_dimension = Narx.pca(train_ae_hidden, pca_accu_var)
    print("The PCA reduced dimension is %d." % reduced_dimension)
    # 计算降维后隐含层
    reduced_train_ae_hidden = np.dot(train_ae_hidden, transfer_matrix)
    # 计算 RVFLNs 输出权重矩阵
    rvflns_hidden_output_w = Narx.rvflns_lsm_calculate(reduced_train_ae_hidden, train_labels)
    if input("Proceed to the ae pca rvflns display?[y/n]") == 'y':
        # 绘图显示
        Display.ae_pca_rvflns_results_display(display_flag, display_start_point, display_num, train_time, test_time,
                                              entry, labels_size, train_features, train_labels,
                                              test_features, test_labels, train_num, test_num,
                                              scaler_labels, rvflns_input_hidden_w, ae_hidden_b, transfer_matrix,
                                              rvflns_hidden_output_w)


def main_test():
    # dataset = range(10)
    # train_dataset, test_dataset = random_split(
    #     dataset=dataset,
    #     lengths=[7, 3],
    # )
    # print(list(train_dataset))
    # print(list(test_dataset))

    # x = torch.rand(100, 10)
    # x_axis = torch.arange(1, 101)
    # for i in range(10):
    #     plt.plot(x_axis, x[:, i], label=f"Column {i + 1}")
    #
    # plt.legend()
    # plt.title("Tensor Column Data")
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.show()

    A = np.random.normal(0, 100, (3, 4))
    print("A:\n{}".format(A))
    y = np.random.normal(0, 100, (3, 5))
    print("y:\n{}".format(y))
    x = Narx.rvflns_lsm_calculate(A, y)
    print("x:\n{}".format(x))
    y_hat = np.dot(A, x)
    print("y_hat:\n{}".format(y_hat))

    # print("input weight matrix calculating...")
    # # 计算 RVFLNs 输入权值矩阵
    # rvflns_input_w = RVFLNs_Assistant.rvflns_input_w_calculate(hidden_reduced, train_features)
    # print("output weight matrix calculating...")
    # # 计算 RVFLNs 输出权值矩阵
    # rvflns_output_w = RVFLNs_Assistant.rvflns_output_w_calculate(hidden_reduced, train_labels)
    # print("\n input weight matrix:")
    # print(rvflns_input_w, "\n")
    # print("output weight matrix:")
    # print(rvflns_output_w, "\n")
    #
    # rvflns_output = np.dot(np.dot(train_features, rvflns_input_w), rvflns_output_w)
    # print(RVFLNs_Assistant.mean_squared_errors(rvflns_output, train_labels))


if __name__ == '__main__':
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
