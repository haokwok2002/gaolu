import numpy as np
import scipy.linalg as la
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import Iter


def pca(x, accu_var):
    # PCA 主成分分析，将数据 N * L 降维为 N * D
    sample_num, features_num = x.shape  # 获取维度
    mean = np.array([np.mean(x[:, i]) for i in range(features_num)])  # 计算每一列的均值
    norm_x = x - mean  # 标准化
    scatter_matrix = np.dot(np.transpose(norm_x), norm_x) / (features_num - 1)  # 计算协方差矩阵
    eig_val, *_, eig_vec = la.eig(scatter_matrix, left=False, right=True)  # 获取特征值和特征向量
    eig_val_abs = np.abs(eig_val)  # 特征值求模

    # 降维维数计算
    eig_val_sum = np.sum(eig_val_abs)  # 特征值求和
    eig_val_var = [eig_val_abs[i] / eig_val_sum for i in range(features_num)]  # 计算方差贡献率
    # 创建一个列表 eig_pairs，其中每个元素是一个元组，包含特征值的方差贡献率和对应的特征向量。
    eig_pairs = [(eig_val_var[i], eig_vec[:, i]) for i in range(features_num)]
    eig_pairs.sort(key=lambda e: e[0], reverse=True)  # 按特征值的方差贡献率从大至小排序

    # 根据累计方差贡献率指标 accu_var 计算降维后的维度 d
    eig_val_accu_var = 0
    d = None
    for var in eig_pairs:
        eig_val_accu_var += var[0]
        if eig_val_accu_var > accu_var:
            d = eig_pairs.index(var)
            break

    feature = np.array([element[1] for element in eig_pairs[:d]])  # 提取前 d 个维度（特征向量）
    return np.transpose(feature.real), d


def self_pinv(x):
    return la.pinv(x)


def rvflns_lsm_calculate(hidden, y):
    w = np.dot(la.pinv(hidden), y)
    return w


def rvflns_lsm_l2_calculate(hidden, y, lam):
    hidden_inv = la.inv(np.dot(np.transpose(hidden), hidden) + lam * np.eye(np.size(hidden, axis=1)))
    w = np.dot(np.dot(hidden_inv, np.transpose(hidden)), y)
    return w


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rvflns_result_display(flag, display_num, train_time, test_time, entry, rvflns_w,
                          reduced_train_hiddens_norm, train_labels_norm, reduced_test_hiddens_norm, test_labels_norm,
                          features_size, labels_size, train_data_length, test_data_length, scaler_labels):
    labels_min = scaler_labels.data_min_
    labels_max = scaler_labels.data_max_
    if flag:
        # 绘制测试集数据
        test_labels_hat_norm = np.dot(reduced_test_hiddens_norm, rvflns_w)

        start_point = random.randint(display_num, test_data_length)
        display_time = test_time[start_point - display_num: start_point]
        display_data = test_labels_norm[start_point - display_num: start_point, :]
        display_data_hat = test_labels_hat_norm[start_point - display_num: start_point, :]
        display_data = scaler_labels.inverse_transform(display_data)
        display_data_hat = scaler_labels.inverse_transform(display_data_hat)
    elif flag == 0:
        # 绘制训练集数据
        train_labels_hat_norm = np.dot(reduced_train_hiddens_norm, rvflns_w)

        start_point = random.randint(display_num, train_data_length)
        display_time = train_time[start_point - display_num: start_point]
        display_data = train_labels_norm[start_point - display_num: start_point, :]
        display_data_hat = train_labels_hat_norm[start_point - display_num: start_point, :]
        display_data = scaler_labels.inverse_transform(display_data)
        display_data_hat = scaler_labels.inverse_transform(display_data_hat)
    else:
        return False

        # 数据显示
    for i in range(labels_size):
        plt.figure(i)
        plt.title("RVFLNs {}".format(entry[features_size - labels_size + i]))
        plt.plot(display_time, display_data[:, i], label='Real', marker='o', markersize=3)
        plt.plot(display_time, display_data_hat[:, i], label='Predicted', marker='o', markersize=3)
        plt.ylim(labels_min[i] - 2, labels_max[i] + 2)
        plt.ylabel(entry[features_size - labels_size + i])
        plt.xlabel('time')
        plt.legend()
        plt.show()

    return True


def mean_squared_errors(results, labels):
    # 如果results为一维，需要将向量形式进行转置
    if results.ndim == 1:
        results = results.reshape(1, results.shape[0])
    batch_size = results.shape[0]
    return 0.5 * np.sum((results - labels) ** 2) / batch_size


def narx_net_test(time_delay, input_hidden_w, hidden_output_w, pca_accu_var, pca_dimension, test_features, test_labels):
    test_loss = 0
    for x, y in Iter.data_iter_narx(time_delay, test_features, test_labels):
        # 计算自编码器的隐藏层
        hidden = np.dot(x.numpy(), input_hidden_w)
        # 计算降维转移矩阵
        transfer_mat, *_ = pca(hidden, pca_accu_var, pca_dimension)
        # 计算降维隐藏层
        hidden_reduced = np.dot(hidden, transfer_mat)
        # 形状改变
        narx_hidden = hidden_reduced.reshape(1, -1)
        # 计算估计值
        y_hat = np.dot(narx_hidden, hidden_output_w)
        # 计算损失
        test_loss += mean_squared_errors(y_hat, y.numpy())

    test_loss = test_loss / (len(test_features) - time_delay)
    return test_loss


def narx_net_train(num_epochs, time_delay, learning_rate, input_hidden_w, hidden_output_w, pca_accu_var, pca_dimension,
                   train_features, train_labels, test_features, test_labels, loss_shut):
    for epoch in range(num_epochs):
        train_loss = np.zeros((pca_dimension, train_labels.size(1)))
        for x, y in Iter.data_iter_narx(time_delay, train_features, train_labels):
            # 计算自编码器的隐藏层
            hidden = np.dot(x.numpy(), input_hidden_w)
            # 计算降维转移矩阵
            transfer_mat, *_ = pca(hidden, pca_accu_var, pca_dimension)
            # 计算降维隐藏层
            hidden_reduced = np.dot(hidden, transfer_mat)
            # 形状改变
            narx_hidden = hidden_reduced.reshape(1, -1)
            # 计算偏差
            delta = np.dot(narx_hidden, hidden_output_w) - y.numpy()
            # 计算损失
            train_loss = -rvflns_lsm_calculate(narx_hidden, delta)
            # 递归
            hidden_output_w += learning_rate * train_loss

        print("epoch: %d, loss: %.4f" % (epoch, np.sum(train_loss)))
        if np.abs(np.sum(train_loss)) < loss_shut:
            print("train loss %.4f < %.4f, train succeed! test..." % (np.sum(train_loss), loss_shut))
            test_loss = narx_net_test(time_delay, input_hidden_w, hidden_output_w, pca_accu_var, pca_dimension,
                                      test_features, test_labels)
            if test_loss < loss_shut:
                print("test loss %.4f < %.4f, test succeed!" % (test_loss, loss_shut))
                if input("Train continue or exit?[t/e]") == 'e':
                    print("Ae train exit.")
                    return True
                else:
                    print("Ae train continue.")
                    continue
            else:
                print("test loss %.4f >= %.4f, test fail, train continue..." % (test_loss, loss_shut))
                continue
        else:
            continue

    narx_net_test(time_delay, input_hidden_w, hidden_output_w, pca_accu_var, pca_dimension,
                  test_features, test_labels)

    print("No train loss < %.4f, train fail." % loss_shut)
    return False


# def narx():
#     # NARX 神经网络训练开始
#     if input("Proceed to the narx net train?[y/n]") == 'y':
#         pass
#     else:
#         print("Program Terminate.")
#         return False
#     # 训练开始
#     narx_input_hidden_w = Narx.self_pinv(ae_net.beta.detach().numpy())
#     narx_hidden_output_w = np.zeros((pca_dimension * time_delay, labels_size))
#     while True:
#         if Narx.narx_net_train(10, time_delay, 1e-5, narx_input_hidden_w, narx_hidden_output_w,
#                                pca_accu_var, pca_dimension, train_features_ts, train_labels_ts,
#                                test_features_ts, test_labels_ts, loss_shut):
#             if input("Save the narx train data?[y/n]") == 'y':
#                 np.save("narx_ih_w.npy", narx_input_hidden_w)
#                 np.save("narx_ho_w.npy", narx_hidden_output_w)
#                 print("Data has saved as 'narx_ih_w.npy' and 'narx_ho_w.npy'.")
#             else:
#                 print("No stored narx train data")
#             break
#         else:
#             answer = input("Train again or save data?[t/s/n]")
#             if answer == 't':
#                 continue
#             elif answer == 's':
#                 np.save("narx_ih_w.npy", narx_input_hidden_w)
#                 np.save("narx_ho_w.npy", narx_hidden_output_w)
#                 print("Data has saved as 'narx_ih_w.npy' and 'narx_ho_w.npy'.")
#                 break
#             else:
#                 print("Ae train failed. The following program is not feasible. Program Terminate.")
#                 return False
