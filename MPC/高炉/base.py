# 库文件
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 设置中文字体
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # 替换为你的中文字体文件路径

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数，用前后两个值的差值按照距离进行加权替换异常值
def replace_outliers_with_weighted_diff(x, y):
    # 计算列的中位数
    median_value = y.median()
    # 检测异常值的索引
    outliers_index = (y - median_value).abs() > 1.5 * y.std()  # 使用标准差作为阈值
    
    # 遍历异常值的索引
    for idx in outliers_index[outliers_index].index:
        # 获取异常值前一个和后一个值的索引
        prev_idx = idx - 1 if idx - 1 >= 0 else idx
        next_idx = idx + 1 if idx + 1 < len(y) else idx
        # 计算当前 x 与前后两个 x 的距离
        dist_prev = abs(x[idx] - x[prev_idx])
        dist_next = abs(x[next_idx] - x[idx])
        total_dist = dist_prev + dist_next
        # 计算权重
        weight_prev = dist_next / total_dist
        weight_next = dist_prev / total_dist
        # 计算前后两个值的差值
        diff = y[next_idx] - y[prev_idx]
        # 根据权重进行插值
        interpolated_value = y[prev_idx] + weight_prev * diff
        # 用插值结果替代异常值
        y[idx] = interpolated_value

# 画出数据
def plot_subplot(data_x,data_y_yuan,data_y,column):
    plt.plot(data_x,data_y_yuan,'r-')
    plt.plot(data_x,data_y,'m-')
    # plt.xlabel(time_term, fontproperties=font)  # 使用中文标签
    plt.ylabel(column, fontproperties=font)  # 使用中文标签
    # 使用中文标签

class MyRNNModel(torch.nn.Module):
    def __init__(self,features_size,hidden_size,isbidirectional):
        super(MyRNNModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=features_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=isbidirectional
        )
        if isbidirectional:
            self.fc = nn.Linear(2 * hidden_size, 2)
        else:
            self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        last_lstm_output = lstm_out[:, -1, :]
        # print(last_lstm_output)
        output = self.fc(last_lstm_output)
        
        return output

    def custom_loss(self, y_true, y_pred):
        squared_diff = torch.pow(y_true - y_pred, 2)
        sum_squared_diff = torch.sum(squared_diff)
        mse = sum_squared_diff / len(y_true)
        return mse



    def my_fit(self, 
                X_train, y_train, 
                X_val, y_val, 
                train_loss_list,val_loss_list,
                epochs=1, batch_size=32, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)


        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                x_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
                y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)

                optimizer.zero_grad()
                y_pred = self(x_batch)
                loss = self.custom_loss(y_batch, y_pred)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            average_epoch_train_loss = epoch_loss / (len(X_train) / batch_size)
            # 验证集评估
            self.eval()
            with torch.no_grad():
                val_loss = 0
                for i in range(0, len(X_val), batch_size):
                    x_batch_val = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32)
                    y_batch_val = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32)

                    y_pred_val = self(x_batch_val)
                    val_loss += self.custom_loss(y_batch_val, y_pred_val).item()

                average_epoch_val_loss = val_loss / (len(X_val) / batch_size)

            print(f'第 {epoch + 1}/{epochs} 轮, 训练误差: {average_epoch_train_loss:.4f}, 验证误差: {average_epoch_val_loss:.4f}', end='\r')
            train_loss_list.append(average_epoch_train_loss)
            val_loss_list.append(average_epoch_val_loss)

        return train_loss_list,val_loss_list
    def my_fit2(self, X_train, y_train):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        x_batch = torch.tensor(X_train, dtype=torch.float32)
        y_batch = torch.tensor(y_train, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = self(x_batch)
        loss = self.custom_loss(y_batch, y_pred)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    

    def my_predict(self, X_test):
        # 设置模型为评估模式，这会关闭 dropout 等层
        self.eval()
        # 将输入数据转换为张量，并设置 requires_grad=True
        x_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
        
        # 获取模型的预测输出
        y_pred = self(x_tensor)
        # 保留预测值的梯度信息
        y_pred.retain_grad()
        # 返回预测结果和包含梯度信息的张量
        return y_pred[:,0].detach().numpy(),y_pred[:,1].detach().numpy()

def double_control_train_test_result(scalers,output_term,
                                    y_test,y_pred_0,y_pred_1,
                                    y_test_2,y_pred_0_2,y_pred_1_2):
    y_test_0 = scalers[output_term[0]].inverse_transform((y_test[:, 0]).reshape(-1, 1)).flatten()
    y_test_1 = scalers[output_term[1]].inverse_transform((y_test[:, 1]).reshape(-1, 1)).flatten()
    y_pred_0_inverse_transform = scalers[output_term[0]].inverse_transform((y_pred_0).reshape(-1, 1)).flatten()
    y_pred_1_inverse_transform = scalers[output_term[1]].inverse_transform((y_pred_1).reshape(-1, 1)).flatten()

    rmse_0 = np.sqrt(mean_squared_error(y_test_0, y_pred_0_inverse_transform))
    rmse_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_1_inverse_transform))

    mre_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0))
    mre_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform) / y_test_1))


    mae_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform)))
    mae_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform)))

    percent0 = np.sum(np.fabs(y_test_0 - y_pred_0_inverse_transform) < 10)/len(y_test_0)
    percent1 = np.sum(np.fabs(y_test_1 - y_pred_1_inverse_transform) < 0.1)/len(y_test_1)
    # 打印结果

    print('训练集')
    print(f"RMSE:  {output_term[0]}: {rmse_0:.4f} , {output_term[1]}: {rmse_1:.4f} ")
    print(f"MAE :  {output_term[0]}: {mae_0:.4f}% , {output_term[1]}: {mae_1:.4f}%")
    print(f"MRE :  {output_term[0]}: { mre_0:.4f}  , {output_term[1]}: { mre_1:.4f} ")
    print(f"per :  {output_term[0]}: { percent0:.4f}  , {output_term[1]}: { percent1:.4f} ")

    # plot_hit_rate_curve(y_test, y_pred_0, y_pred_1)


    output0 = y_test_0 - y_pred_0_inverse_transform
    output1 = y_test_1 - y_pred_1_inverse_transform

    # print(f"误差分析0:平均值:{output0.std():.4f},方差:{output0.mean():.4f}")
    # print(f"误差分析1:平均值:{output1.std():.4f},方差:{output1.mean():.4f}")

    plt.figure(figsize=(8, 6))
    plt.subplot(4, 2, 1)
    plt.plot(y_test_0,'r')
    plt.plot(y_pred_0_inverse_transform,'g')
    plt.ylabel(output_term[0], fontproperties=font)  # 使用中文标签
    plt.title("建模效果", fontproperties=font)

    plt.subplot(4, 2, 3)
    plt.plot(y_test_1,'r')
    plt.plot(y_pred_1_inverse_transform,'g')
    plt.ylabel(output_term[1], fontproperties=font)  # 使用中文标签

    plt.subplot(4, 2, 5)
    plt.plot(output0,'r-')
    plt.ylabel(output_term[0]+'_err', fontproperties=font)  # 使用中文标签

    plt.subplot(4, 2, 7)
    plt.plot(output1,'r-')
    plt.ylabel(output_term[1]+'_err', fontproperties=font)  # 使用中文标签




    y_test_0 = scalers[output_term[0]].inverse_transform((y_test_2[:, 0]).reshape(-1, 1)).flatten()
    y_test_1 = scalers[output_term[1]].inverse_transform((y_test_2[:, 1]).reshape(-1, 1)).flatten()
    y_pred_0_inverse_transform = scalers[output_term[0]].inverse_transform((y_pred_0_2).reshape(-1, 1)).flatten()
    y_pred_1_inverse_transform = scalers[output_term[1]].inverse_transform((y_pred_1_2).reshape(-1, 1)).flatten()

    rmse_0 = np.sqrt(mean_squared_error(y_test_0, y_pred_0_inverse_transform))
    rmse_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_1_inverse_transform))

    # 计算 
    mre_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0))
    mre_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform) / y_test_1))


    mae_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform)))
    mae_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform)))

    percent0 = np.sum(np.fabs(y_test_0 - y_pred_0_inverse_transform) < 10)/len(y_test_0)
    percent1 = np.sum(np.fabs(y_test_1 - y_pred_1_inverse_transform) < 0.1)/len(y_test_1)
    # 打印结果
    print('测试集')
    print(f"RMSE:  {output_term[0]}: {rmse_0:.4f} , {output_term[1]}: {rmse_1:.4f} ")
    print(f"MAE :  {output_term[0]}: {mae_0:.4f}% , {output_term[1]}: {mae_1:.4f}%")
    print(f"MRE :  {output_term[0]}: { mre_0:.4f}  , {output_term[1]}: { mre_1:.4f} ")
    print(f"per :  {output_term[0]}: { percent0:.4f}  , {output_term[1]}: { percent1:.4f} ")

    # plot_hit_rate_curve(y_test, y_pred_0, y_pred_1)


    output0 = y_test_0 - y_pred_0_inverse_transform
    output1 = y_test_1 - y_pred_1_inverse_transform

    # print(f"误差分析0:平均值:{output0.std():.4f},方差:{output0.mean():.4f}")
    # print(f"误差分析1:平均值:{output1.std():.4f},方差:{output1.mean():.4f}")

    plt.subplot(4, 2, 2)
    plt.plot(y_test_0,'r', label="真实值")
    plt.plot(y_pred_0_inverse_transform,'g', label="预测值")
    plt.ylabel(output_term[0], fontproperties=font)  # 使用中文标签
    plt.legend(prop=font)  # 添加图例并设置字体为中文
    plt.title("预测效果", fontproperties=font)

    plt.subplot(4, 2, 4)
    plt.plot(y_test_1,'r')
    plt.plot(y_pred_1_inverse_transform,'g')
    plt.ylabel(output_term[1], fontproperties=font)  # 使用中文标签

    plt.subplot(4, 2, 6)
    plt.plot(output0,'r-')
    plt.ylabel(output_term[0]+'_err', fontproperties=font)  # 使用中文标签

    plt.subplot(4, 2, 8)
    plt.plot(output1,'r-')
    plt.ylabel(output_term[1]+'_err', fontproperties=font)  # 使用中文标签



    plt.tight_layout()
    plt.show()



    
def double_control_train_test_result_data(scalers,output_term,
                                    y_test,y_pred_0,y_pred_1,
                                    y_test_2,y_pred_0_2,y_pred_1_2):
    y_test_0 = scalers[output_term[0]].inverse_transform((y_test[:, 0]).reshape(-1, 1)).flatten()
    y_test_1 = scalers[output_term[1]].inverse_transform((y_test[:, 1]).reshape(-1, 1)).flatten()
    y_pred_0_inverse_transform = scalers[output_term[0]].inverse_transform((y_pred_0).reshape(-1, 1)).flatten()
    y_pred_1_inverse_transform = scalers[output_term[1]].inverse_transform((y_pred_1).reshape(-1, 1)).flatten()

    rmse_0 = np.sqrt(mean_squared_error(y_test_0, y_pred_0_inverse_transform))
    rmse_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_1_inverse_transform))

    # 计算 
    mre_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0))
    mre_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform) / y_test_1))


    mape_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0)) * 100
    mape_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform) / y_test_1)) * 100

    percent0 = np.sum(np.fabs(y_test_0 - y_pred_0_inverse_transform) < 10)/len(y_test_0)
    percent1 = np.sum(np.fabs(y_test_1 - y_pred_1_inverse_transform) < 0.1)/len(y_test_1)
    # 打印结果

    print('训练集')
    print(f"RMSE:  {output_term[0]}: {rmse_0:.4f} , {output_term[1]}: {rmse_1:.4f} ")
    print(f"MAPE:  {output_term[0]}: {mape_0:.4f}% , {output_term[1]}: {mape_1:.4f}%")
    print(f"MRE :  {output_term[0]}: { mre_0:.4f}  , {output_term[1]}: { mre_1:.4f} ")
    print(f"per :  {output_term[0]}: { percent0:.4f}  , {output_term[1]}: { percent1:.4f} ")

    # plot_hit_rate_curve(y_test, y_pred_0, y_pred_1)


    output0 = y_test_0 - y_pred_0_inverse_transform
    output1 = y_test_1 - y_pred_1_inverse_transform

    # print(f"误差分析0:平均值:{output0.std():.4f},方差:{output0.mean():.4f}")
    # print(f"误差分析1:平均值:{output1.std():.4f},方差:{output1.mean():.4f}")





    y_test_0 = scalers[output_term[0]].inverse_transform((y_test_2[:, 0]).reshape(-1, 1)).flatten()
    y_test_1 = scalers[output_term[1]].inverse_transform((y_test_2[:, 1]).reshape(-1, 1)).flatten()
    y_pred_0_inverse_transform = scalers[output_term[0]].inverse_transform((y_pred_0_2).reshape(-1, 1)).flatten()
    y_pred_1_inverse_transform = scalers[output_term[1]].inverse_transform((y_pred_1_2).reshape(-1, 1)).flatten()

    rmse_0 = np.sqrt(mean_squared_error(y_test_0, y_pred_0_inverse_transform))
    rmse_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_1_inverse_transform))

    # 计算 
    mre_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0))
    mre_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform) / y_test_1))


    mape_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0)) * 100
    mape_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform) / y_test_1)) * 100

    percent0 = np.sum(np.fabs(y_test_0 - y_pred_0_inverse_transform) < 10)/len(y_test_0)
    percent1 = np.sum(np.fabs(y_test_1 - y_pred_1_inverse_transform) < 0.1)/len(y_test_1)
    # 打印结果
    print('测试集')
    print(f"RMSE:  {output_term[0]}: {rmse_0:.4f} , {output_term[1]}: {rmse_1:.4f} ")
    print(f"MAPE:  {output_term[0]}: {mape_0:.4f}% , {output_term[1]}: {mape_1:.4f}%")
    print(f"MRE :  {output_term[0]}: { mre_0:.4f}  , {output_term[1]}: { mre_1:.4f} ")
    print(f"per :  {output_term[0]}: { percent0:.4f}  , {output_term[1]}: { percent1:.4f} ---")

    # plot_hit_rate_curve(y_test, y_pred_0, y_pred_1)


    output0 = y_test_0 - y_pred_0_inverse_transform
    output1 = y_test_1 - y_pred_1_inverse_transform

    # print(f"误差分析0:平均值:{output0.std():.4f},方差:{output0.mean():.4f}")
    # print(f"误差分析1:平均值:{output1.std():.4f},方差:{output1.mean():.4f}")



def double_control_predict_result(scalers,output_term,y_test,y_pred_0,y_pred_1):
    y_test_0 = scalers[output_term[0]].inverse_transform((y_test[:, 0]).reshape(-1, 1)).flatten()
    y_test_1 = scalers[output_term[1]].inverse_transform((y_test[:, 1]).reshape(-1, 1)).flatten()
    y_pred_0_inverse_transform = scalers[output_term[0]].inverse_transform((y_pred_0).reshape(-1, 1)).flatten()
    y_pred_1_inverse_transform = scalers[output_term[1]].inverse_transform((y_pred_1).reshape(-1, 1)).flatten()

    rmse_0 = np.sqrt(mean_squared_error(y_test_0, y_pred_0_inverse_transform))
    rmse_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_1_inverse_transform))

    # 计算 
    mre_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0))
    mre_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform) / y_test_1))


    mape_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0)) * 100
    mape_1 = np.mean(np.abs((y_test_1 - y_pred_1_inverse_transform) / y_test_1)) * 100

    percent0 = np.sum(np.fabs(y_test_0 - y_pred_0_inverse_transform) < 10)/len(y_test_0)
    percent1 = np.sum(np.fabs(y_test_1 - y_pred_1_inverse_transform) < 0.1)/len(y_test_1)
    # 打印结果

    print(f"RMSE:  {output_term[0]}: {rmse_0:.4f} , {output_term[1]}: {rmse_1:.4f} ")
    print(f"MAPE:  {output_term[0]}: {mape_0:.4f}% , {output_term[1]}: {mape_1:.4f}%")
    print(f"MRE :  {output_term[0]}: { mre_0:.4f}  , {output_term[1]}: { mre_1:.4f} ")
    print(f"per :  {output_term[0]}: { percent0:.4f}  , {output_term[1]}: { percent1:.4f} ")

    # plot_hit_rate_curve(y_test, y_pred_0, y_pred_1)


    output0 = y_test_0 - y_pred_0_inverse_transform
    output1 = y_test_1 - y_pred_1_inverse_transform

    # print(f"误差分析0:平均值:{output0.std():.4f},方差:{output0.mean():.4f}")
    # print(f"误差分析1:平均值:{output1.std():.4f},方差:{output1.mean():.4f}")

    plt.subplot(4, 1, 1)
    plt.plot(y_test_0,'r')
    plt.plot(y_pred_0_inverse_transform,'g')
    plt.ylabel(output_term[0], fontproperties=font)  # 使用中文标签

    plt.subplot(4, 1, 2)
    plt.plot(y_test_1,'r')
    plt.plot(y_pred_1_inverse_transform,'g')
    plt.ylabel(output_term[1], fontproperties=font)  # 使用中文标签

    plt.subplot(4, 1, 3)
    plt.plot(output0,'k')
    plt.ylabel(output_term[0]+'_err', fontproperties=font)  # 使用中文标签

    plt.subplot(4, 1, 4)
    plt.plot(output1,'k')
    plt.ylabel(output_term[1]+'_err', fontproperties=font)  # 使用中文标签

    plt.tight_layout()
    plt.show()

def single_control_predict_result(scalers,output_term,y_test,y_pred_0):
    
    y_test_0 = scalers[output_term[0]].inverse_transform((y_test.reshape(-1, 1))).flatten()
    y_pred_0_inverse_transform = scalers[output_term[0]].inverse_transform((y_pred_0.reshape(-1, 1))).flatten()

    rmse_0 = np.sqrt(mean_squared_error(y_test_0, y_pred_0_inverse_transform))

    # 计算 
    mre_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0))

    # 打印结果
    print(f"RMSE: {output_term[0]}: {rmse_0:.4f}")
    print(f"MRE : {output_term[0]}: { mre_0:.4f}")
    mape_0 = np.mean(np.abs((y_test_0 - y_pred_0_inverse_transform) / y_test_0)) * 100
    print(f"MAPE: {output_term[0]}: {mape_0:.4f}%")



    output0 = y_test_0 - y_pred_0_inverse_transform
    MAE = np.mean(np.abs(output0))
    RMSE = np.sqrt(MAE)
    HR = np.sum(np.abs(output0)<0.1)/len(output0)
    RE = np.sum(output0**2)/np.sum(y_test**2)
    print(f'MAE:{MAE:.4f},RMSE:{RMSE:.4f},HR:{HR:.4f},RE:{RE:.4f}')

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(y_test_0,'r')
    plt.plot(y_pred_0_inverse_transform,'g')
    plt.ylabel(output_term[0], fontproperties=font)  # 使用中文标签



    plt.subplot(2, 1, 2)
    plt.plot(output0,'r-')
    plt.ylabel(output_term[0]+'_err', fontproperties=font)  # 使用中文标签
    plt.suptitle('高炉模型预测结果', fontproperties=font)  # 添加整个图形的标题
    plt.tight_layout()
    plt.show()

def gaolu_predict_raw(scalers,output_term,model,model_gaolu,X_predict_test,y_predict_test):
    y_test = y_predict_test
    y_test_0 = scalers[output_term[0]].inverse_transform(y_test[:,0].reshape(-1, 1)).flatten()
    y_test_1 = scalers[output_term[1]].inverse_transform(y_test[:,1].reshape(-1, 1)).flatten()
    y_pred_0,y_pred_1  = model.my_predict(X_predict_test)
    y_pred_0_predict = scalers[output_term[0]].inverse_transform((y_pred_0).reshape(-1, 1)).flatten()
    y_pred_1_predict = scalers[output_term[1]].inverse_transform((y_pred_1).reshape(-1, 1)).flatten()

    y_pred_0,y_pred_1  = model_gaolu.my_predict(X_predict_test)
    y_pred_0_gaolu = scalers[output_term[0]].inverse_transform((y_pred_0).reshape(-1, 1)).flatten()
    y_pred_1_gaolu = scalers[output_term[1]].inverse_transform((y_pred_1).reshape(-1, 1)).flatten()

    plt.subplot(2, 1, 1)
    plt.plot(y_test_0, 'r-', label='实际值')
    plt.plot(y_pred_0_predict, 'go-', label='预测模型')
    plt.plot(y_pred_0_gaolu, 'b', label='高炉模型')
    plt.legend(prop=font)
    plt.subplot(2, 1, 2)
    plt.plot(y_test_1, 'r-')
    plt.plot(y_pred_1_predict, 'go-')
    plt.plot(y_pred_1_gaolu, 'b')

def gaolu_predict_raw_gggggg(scalers,output_term,model,model_gaolu,X_predict_test,y_predict_test):
    y_test = y_predict_test[:-1]

    y_test_0 = scalers[output_term[0]].inverse_transform(y_test[:,0].reshape(-1, 1)).flatten()
    y_test_1 = scalers[output_term[1]].inverse_transform(y_test[:,1].reshape(-1, 1)).flatten()
    y_pred_0,y_pred_1  = model.my_predict(X_predict_test)
    y_pred_0_predict = scalers[output_term[0]].inverse_transform((y_pred_0).reshape(-1, 1)).flatten()[1:]
    y_pred_1_predict = scalers[output_term[1]].inverse_transform((y_pred_1).reshape(-1, 1)).flatten()[1:]
    y_pred_0,y_pred_1  = model_gaolu.my_predict(X_predict_test)
    y_pred_0_gaolu = scalers[output_term[0]].inverse_transform((y_pred_0).reshape(-1, 1)).flatten()[1:]
    y_pred_1_gaolu = scalers[output_term[1]].inverse_transform((y_pred_1).reshape(-1, 1)).flatten()[1:]

    plt.subplot(2, 1, 1)
    plt.plot(y_test_0, 'r-', label='实际值')
    plt.plot(y_pred_0_predict, 'go-', label='预测模型')
    plt.plot(y_pred_0_gaolu, 'b', label='高炉模型')
    plt.legend(prop=font)
    plt.subplot(2, 1, 2)
    plt.plot(y_test_1, 'r-')
    plt.plot(y_pred_1_predict, 'go-')
    plt.plot(y_pred_1_gaolu, 'b')

class CustomPredictor_double:
    def __init__(self, model, hidden_size):
        self.model = model
        self.hidden_size = hidden_size
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def lstm_forward(self, input, initial_states, w_ih, w_hh, b_ih, b_hh):
        h_0, c_0 = initial_states  # 初始状态  [b_size, hidden_size]
        b_size, seq_len, input_size = input.shape
        h_size = h_0.shape[-1]

        h_prev, c_prev = h_0, c_0

        # 使用 np.newaxis 在第一个维度上插入一个新的维度  # 使用 np.tile 在第一个维度上复制 b_size 次
        w_ih_expanded = w_ih[np.newaxis, :, :]    
        w_ih_batch = np.tile(w_ih_expanded, (b_size, 1, 1))
        w_hh_expanded = w_hh[np.newaxis, :, :]    
        w_hh_batch = np.tile(w_hh_expanded, (b_size, 1, 1))
        # print(w_ih_batch.shape)

        output_size = h_size
        output = np.zeros((b_size, seq_len, output_size))  # 初始化一个输出序列
        for t in range(seq_len):
            x = input[:, t, :]  # 当前时刻的输入向量 [b,in_size]->[b,in_size,1]
            w_times_x = np.matmul(w_ih_batch, x[:, :, np.newaxis]).squeeze(-1)   # bmm:含有批量大小的矩阵相乘
            # [b, 4*hidden_size, 1]->[b, 4*hidden_size]
            # 这一步就是计算了 Wii*xt|Wif*xt|Wig*xt|Wio*xt
            w_times_h_prev = np.matmul(w_hh_batch, h_prev[:, :, np.newaxis]).squeeze(-1)
            # [b, 4*hidden_size, hidden_size]*[b, hidden_size, 1]->[b,4*hidden_size, 1]->[b, 4*hidden_size]
            # 这一步就是计算了 Whi*ht-1|Whf*ht-1|Whg*ht-1|Who*ht-1

            # 分别计算输入门(i)、遗忘门(f)、cell门(g)、输出门(o)  维度均为 [b, h_size]
            i_t = self.sigmoid(w_times_x[:, :h_size] + w_times_h_prev[:, :h_size] + b_ih[:h_size] + b_hh[:h_size])  # 取前四分之一
            f_t = self.sigmoid(w_times_x[:, h_size:2*h_size] + w_times_h_prev[:, h_size:2*h_size]
                                + b_ih[h_size:2*h_size] + b_hh[h_size:2*h_size])
            g_t = np.tanh(w_times_x[:, 2*h_size:3*h_size] + w_times_h_prev[:, 2*h_size:3*h_size]
                                + b_ih[2*h_size:3*h_size] + b_hh[2*h_size:3*h_size])
            o_t = self.sigmoid(w_times_x[:, 3*h_size:] + w_times_h_prev[:, 3*h_size:]
                                + b_ih[3*h_size:] + b_hh[3*h_size:])
            c_prev = f_t * c_prev + i_t * g_t
            h_prev = o_t * np.tanh(c_prev)

            output[:, t, :] = h_prev

        return output, (np.expand_dims(h_prev, axis=0), np.expand_dims(c_prev, axis=0))  # 官方是三维，在第0维扩一维

    def predict(self, data_input):

        input = data_input  # 随机初始化一个输入序列
        c_0 = np.zeros((data_input.shape[0], self.hidden_size))  # 初始值，不会参与训练
        h_0 = np.zeros((data_input.shape[0], self.hidden_size))

        output_forward, (h_n_me, c_n_me) = self.lstm_forward(input, (h_0, c_0), 
                                                    self.model.lstm.weight_ih_l0.detach().numpy(),
                                                    self.model.lstm.weight_hh_l0.detach().numpy(), 
                                                    self.model.lstm.bias_ih_l0.detach().numpy(), 
                                                    self.model.lstm.bias_hh_l0.detach().numpy())

        last_lstm_output_forward = output_forward[:, -1, :]

        output_backward, (h_n_me, c_n_me) = self.lstm_forward(input, (h_0, c_0), 
                                                    self.model.lstm.weight_ih_l0_reverse.detach().numpy(),
                                                    self.model.lstm.weight_hh_l0_reverse.detach().numpy(), 
                                                    self.model.lstm.bias_ih_l0_reverse.detach().numpy(), 
                                                    self.model.lstm.bias_hh_l0_reverse.detach().numpy())

        last_lstm_output_backward = output_backward[:, -1, :]
        # print(last_lstm_output_forward.shape)
        # print(last_lstm_output_backward.shape)
        # 最终输出
        combined_hidden = np.concatenate((last_lstm_output_forward, last_lstm_output_backward), axis=1)
        # print(combined_hidden.shape)


        output = (np.dot(combined_hidden, np.transpose(self.model.fc.weight.detach().numpy()))
                    + self.model.fc.bias.detach().numpy()
    )
        y_pred_0, y_pred_1= output[:,0],output[:,1]

        # y_pred_0 = scalers[output_term[0]].inverse_transform(np.array(y_pred_0).reshape(-1, 1)).flatten()
        # y_pred_1 = scalers[output_term[1]].inverse_transform(np.array(y_pred_1).reshape(-1, 1)).flatten()




        return y_pred_0, y_pred_1

# 生成期望数据
def get_y_aim_data(scalers,output_term,Times):
    set_y1 = np.full(Times,1500)
    set_y1[30:] = 1510
    set_y1[60:] = 1505
    set_y1[90:] = 1515
    # set_y1[70:] = 1520
    # set_y1[90:] = 1525

    set_y2 = np.full(Times,0.44)
    set_y2[15:] = 0.48
    set_y2[45:] = 0.52
    set_y2[75:] = 0.44
    # 限制设定值在 -1 到 1 之间
    # set_y1 = np.clip(set_y1, -1, 1)
    # set_y2 = np.clip(set_y2, -1, 1)

    set_y1_trans = scalers[output_term[0]].transform(set_y1.reshape(-1,1)).flatten()
    set_y2_trans = scalers[output_term[1]].transform(set_y2.reshape(-1,1)).flatten()

    return set_y1, set_y2, set_y1_trans, set_y2_trans

# 生成参考轨迹
def generate_yr(aim_value,current_value,alpha,P):
    # 生成设定信号
    setpoint_signal = np.full(10, aim_value)
    # 初始化参数
    alpha = alpha
    y_r = np.zeros(P)
    y_r[0] = current_value
    # 模拟一阶模型
    for k in range(1,P):
        y_r[k] = alpha * y_r[k-1] + (1 - alpha) * aim_value

    # # 绘制结果
    # plt.plot(setpoint_signal, label='Setpoint Signal')
    # plt.plot(y_r,'o-', label='Output Signal (Tracked)')
    # plt.legend()
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Tracking Setpoint Signal with One-Order Model')
    # plt.show()
    return y_r

def data_tranform_plot_7_2(scalers,Times,max_control,
                        output_term,input_term,
                        set_y1,set_y2,set_y1_trans,set_y2_trans,
                        all_pred_y1, all_pred_y2,
                        all_pred_u1,
                        all_pred_u2,
                        all_pred_u3,
                        all_pred_u4,
                        all_pred_u5,
                        all_pred_u6,
                        all_pred_u7):
    y1_pred_inverse_transform = scalers[output_term[0]].inverse_transform(np.array(all_pred_y1).reshape(-1, 1)).flatten()
    y2_pred_inverse_transform = scalers[output_term[1]].inverse_transform(np.array(all_pred_y2).reshape(-1, 1)).flatten()
    all_pred_u1_inverse_transform = scalers[input_term[0]].inverse_transform(np.array(all_pred_u1).reshape(-1, 1)).flatten()
    all_pred_u2_inverse_transform = scalers[input_term[1]].inverse_transform(np.array(all_pred_u2).reshape(-1, 1)).flatten()
    all_pred_u3_inverse_transform = scalers[input_term[2]].inverse_transform(np.array(all_pred_u3).reshape(-1, 1)).flatten()
    all_pred_u4_inverse_transform = scalers[input_term[3]].inverse_transform(np.array(all_pred_u4).reshape(-1, 1)).flatten()
    all_pred_u5_inverse_transform = scalers[input_term[4]].inverse_transform(np.array(all_pred_u2).reshape(-1, 1)).flatten()
    all_pred_u6_inverse_transform = scalers[input_term[5]].inverse_transform(np.array(all_pred_u3).reshape(-1, 1)).flatten()
    all_pred_u7_inverse_transform = scalers[input_term[6]].inverse_transform(np.array(all_pred_u4).reshape(-1, 1)).flatten()
    a1 = scalers[input_term[0]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a2 = scalers[input_term[1]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a3 = scalers[input_term[2]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a4 = scalers[input_term[3]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a5 = scalers[input_term[4]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a6 = scalers[input_term[5]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a7 = scalers[input_term[6]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    print(f'上线分别是：{a1}、{a2}、{a3}、{a4}、{a5}、{a6}、{a7}')


    rmse_1 = np.mean(np.fabs(set_y1-y1_pred_inverse_transform))
    rmse_2 = np.mean(np.fabs(set_y2-y2_pred_inverse_transform))
    print('平均误差',rmse_1.round(4))
    print('平均误差',rmse_2.round(4))

    # 模型预测控制结果可视化
    # 创建两个子图，分别绘制每个维度
    plt.figure(figsize=(14, 10))

    # 第一个维度的曲线
    plt.subplot(9, 2, 1)
    plt.plot(set_y1_trans, 'ro-', label='设定值')
    plt.plot(all_pred_y1, 'bo-', label='实际值')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylabel(output_term[0], fontproperties=font)  # 使用中文标签
    plt.legend(prop=font)
    plt.title("归一化", fontproperties=font)
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第二个维度的曲线
    plt.subplot(9, 2, 3)
    plt.plot(set_y2_trans, 'ro-')
    plt.plot(all_pred_y2, 'bo-')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylabel(output_term[1], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第一个维度的u1曲线
    plt.subplot(9, 2, 5)
    plt.plot(all_pred_u1, 'bo-', label='u1')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[0], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第二个维度的u2曲线
    plt.subplot(9, 2, 7)
    plt.plot(all_pred_u2, 'bo-', label='u2')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[1], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第三个维度的u3曲线
    plt.subplot(9, 2, 9)
    plt.plot(all_pred_u3, 'bo-', label='u3')  # 修改标签为 'u3'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[2], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第四个维度的u4曲线
    plt.subplot(9, 2, 11)
    plt.plot(all_pred_u4, 'bo-', label='u4')  # 修改标签为 'u4'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[3], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')


    # 第二个维度的u2曲线
    plt.subplot(9, 2, 13)
    plt.plot(all_pred_u5, 'bo-', label='u5')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[4], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第三个维度的u3曲线
    plt.subplot(9, 2, 15)
    plt.plot(all_pred_u6, 'bo-', label='u6')  # 修改标签为 'u3'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[5], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第四个维度的u4曲线
    plt.subplot(9, 2, 17)
    plt.plot(all_pred_u7, 'bo-', label='u7')  # 修改标签为 'u4'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[6], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')



    ######################################################

    # 第一个维度的曲线
    plt.subplot(9, 2, 2)
    plt.plot(set_y1, 'ro-')
    plt.plot(y1_pred_inverse_transform, 'bo-')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylabel(output_term[0], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.title("正常", fontproperties=font)
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第二个维度的曲线
    plt.subplot(9, 2, 4)
    plt.plot(set_y2, 'ro-')
    plt.plot(y2_pred_inverse_transform, 'bo-')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylabel(output_term[1], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第一个维度的u1曲线
    plt.subplot(9, 2, 6)
    plt.plot(all_pred_u1_inverse_transform, 'bo-', label='u1')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a1[1],a1[0]))
    plt.ylabel(input_term[0], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第二个维度的u2曲线
    plt.subplot(9, 2, 8)
    plt.plot(all_pred_u2_inverse_transform, 'bo-', label='u2')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a2[1],a2[0]))
    plt.ylabel(input_term[1], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第三个维度的u3曲线
    plt.subplot(9, 2, 10)
    plt.plot(all_pred_u3_inverse_transform, 'bo-', label='u3')  # 修改标签为 'u3'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a3[1],a3[0]))
    plt.ylabel(input_term[2], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第四个维度的u4曲线
    plt.subplot(9, 2, 12)
    plt.plot(all_pred_u4_inverse_transform, 'bo-', label='u4')  # 修改标签为 'u4'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a4[1],a4[0]))
    plt.ylabel(input_term[3], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')


    # 第二个维度的u5曲线
    plt.subplot(9, 2, 14)
    plt.plot(all_pred_u5_inverse_transform, 'bo-', label='u2')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a5[1],a5[0]))
    plt.ylabel(input_term[4], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第三个维度的u6曲线
    plt.subplot(9, 2, 16)
    plt.plot(all_pred_u6_inverse_transform, 'bo-', label='u3')  # 修改标签为 'u3'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a6[1],a6[0]))
    plt.ylabel(input_term[5], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第四个维度的u7曲线
    plt.subplot(9, 2, 18)
    plt.plot(all_pred_u7_inverse_transform, 'bo-', label='u4')  # 修改标签为 'u4'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a7[1],a7[0]))
    plt.ylabel(input_term[6], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')


    # 调整子图布局
    plt.tight_layout()
    plt.show()











def data_tranform_plot_4_2(scalers,Times,max_control,
                        output_term,input_term,
                        set_y1,set_y2,set_y1_trans,set_y2_trans,
                        all_pred_y1, all_pred_y2,
                        all_pred_u1,
                        all_pred_u2,
                        all_pred_u3,
                        all_pred_u4):
    y1_pred_inverse_transform = scalers[output_term[0]].inverse_transform(np.array(all_pred_y1).reshape(-1, 1)).flatten()
    y2_pred_inverse_transform = scalers[output_term[1]].inverse_transform(np.array(all_pred_y2).reshape(-1, 1)).flatten()
    all_pred_u1_inverse_transform = scalers[input_term[0]].inverse_transform(np.array(all_pred_u1).reshape(-1, 1)).flatten()
    all_pred_u2_inverse_transform = scalers[input_term[1]].inverse_transform(np.array(all_pred_u2).reshape(-1, 1)).flatten()
    all_pred_u3_inverse_transform = scalers[input_term[2]].inverse_transform(np.array(all_pred_u3).reshape(-1, 1)).flatten()
    all_pred_u4_inverse_transform = scalers[input_term[3]].inverse_transform(np.array(all_pred_u4).reshape(-1, 1)).flatten()
    a1 = scalers[input_term[0]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a2 = scalers[input_term[1]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a3 = scalers[input_term[2]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a4 = scalers[input_term[3]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    print(f'上线分别是：{a1}、{a2}、{a3}、{a4}')


    rmse_1 = np.mean(np.fabs(set_y1-y1_pred_inverse_transform))
    rmse_2 = np.mean(np.fabs(set_y2-y2_pred_inverse_transform))
    print('平均误差',rmse_1.round(4))
    print('平均误差',rmse_2.round(4))

    # 模型预测控制结果可视化
    # 创建两个子图，分别绘制每个维度
    plt.figure(figsize=(14, 10))

    # 第一个维度的曲线
    plt.subplot(6, 2, 1)
    plt.plot(set_y1_trans, 'ro-', label='设定值')
    plt.plot(all_pred_y1, 'bo-', label='实际值')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylabel(output_term[0], fontproperties=font)  # 使用中文标签
    plt.legend(prop=font)
    plt.title("归一化", fontproperties=font)
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第二个维度的曲线
    plt.subplot(6, 2, 3)
    plt.plot(set_y2_trans, 'ro-')
    plt.plot(all_pred_y2, 'bo-')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylabel(output_term[1], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第一个维度的u1曲线
    plt.subplot(6, 2, 5)
    plt.plot(all_pred_u1, 'bo-', label='u1')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[0], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第二个维度的u2曲线
    plt.subplot(6, 2, 7)
    plt.plot(all_pred_u2, 'bo-', label='u2')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[1], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第三个维度的u3曲线
    plt.subplot(6, 2, 9)
    plt.plot(all_pred_u3, 'bo-', label='u3')  # 修改标签为 'u3'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[2], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第四个维度的u4曲线
    plt.subplot(6, 2, 11)
    plt.plot(all_pred_u4, 'bo-', label='u4')  # 修改标签为 'u4'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylim((-max_control,max_control))
    plt.ylabel(input_term[3], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')



    ######################################################

    # 第一个维度的曲线
    plt.subplot(6, 2, 2)
    plt.plot(set_y1, 'ro-')
    plt.plot(y1_pred_inverse_transform, 'bo-')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylabel(output_term[0], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.title("正常", fontproperties=font)
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第二个维度的曲线
    plt.subplot(6, 2, 4)
    plt.plot(set_y2, 'ro-')
    plt.plot(y2_pred_inverse_transform, 'bo-')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    plt.ylabel(output_term[1], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第一个维度的u1曲线
    plt.subplot(6, 2, 6)
    plt.plot(all_pred_u1_inverse_transform, 'bo-', label='u1')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a1[1],a1[0]))
    plt.ylabel(input_term[0], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第二个维度的u2曲线
    plt.subplot(6, 2, 8)
    plt.plot(all_pred_u2_inverse_transform, 'bo-', label='u2')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a2[1],a2[0]))
    plt.ylabel(input_term[1], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第三个维度的u3曲线
    plt.subplot(6, 2, 10)
    plt.plot(all_pred_u3_inverse_transform, 'bo-', label='u3')  # 修改标签为 'u3'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a3[1],a3[0]))
    plt.ylabel(input_term[2], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')

    # 第四个维度的u4曲线
    plt.subplot(6, 2, 12)
    plt.plot(all_pred_u4_inverse_transform, 'bo-', label='u4')  # 修改标签为 'u4'
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlim((0,Times))
    # plt.ylim((a4[1],a4[0]))
    plt.ylabel(input_term[3], fontproperties=font)  # 使用中文标签
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7, color='gray')



    # 调整子图布局
    plt.tight_layout()
    plt.show()












































