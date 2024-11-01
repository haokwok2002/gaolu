print('加载库文件')
# 基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
import datetime
import pickle
# 机器学习库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
# 优化相关库
from skopt import gp_minimize
from scipy.optimize import minimize
# 深度学习库
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
# 自定义模块
import base
# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# 中文字体设置
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # 替换为你的中文字体文件路径
# 其他路径设置
sys.path.append(r"C:\Users\haokw\Documents\GitHub\gaolu\MPC\高炉")




# 读取Excel文件
excel_path = f'C:\\Users\\haokw\\Documents\\GitHub\\gaolu\\MPC\\高炉\\0数据处理\\新输入输出模式\\1h_mean.xlsx'
df_sheet_yuansu = pd.read_excel(excel_path, sheet_name='原始输出') 

excel_path = f'C:\\Users\\haokw\\Documents\\GitHub\\gaolu\\MPC\\高炉\\0数据处理\\新输入输出模式\\1h_mean.xlsx'
df_sheet_params = pd.read_excel(excel_path, sheet_name='1h_mean_all') 



input_term          = ['富氧流量', '冷风流量', '热风压力', '热风温度']
output_term         = ['铁水温度[MIT]', '铁水硅含量[SI]']
last_output_term    = ['铁水温度[MIT]2', '铁水硅含量[SI]2']
time_term           = '时间戳h'
