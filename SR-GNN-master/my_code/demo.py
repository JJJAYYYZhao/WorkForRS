import torch
import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


percent=0.85

train_data = pickle.load(open('../datasets/' + "sample" + '/train.txt', 'rb'))
interval=train_data[2]
sorted_interval=sorted(list(int(reduce(lambda x, y: x + y, interval[i][j:]))for i in range(len(interval)) for j in range(len(interval[i]))))

print("85%:"+str(sorted_interval[int(len(sorted_interval)*percent)-1]))

print("min:"+str(sorted_interval[0]))
print("max:"+str(sorted_interval[int(len(sorted_interval))-1]))

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# 随机生成（10000,）服从正态分布的数据
data = sorted_interval
"""
绘制直方图
data:必选参数，绘图数据
bins:直方图的长条形数目，可选项，默认为10
normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
facecolor:长条形的颜色
edgecolor:长条形边框的颜色
alpha:透明度
"""
plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("时间片")
# 显示纵轴标签
plt.ylabel("频率")
# 显示图标题
plt.title("频数/频率分布直方图")
plt.show()