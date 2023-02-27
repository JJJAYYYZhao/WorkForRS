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
# total_interval=[]
# for item in interval:
#     total_interval+=item
# sorted_interval=sorted(total_interval)
print("85%:"+str(sorted_interval[int(len(sorted_interval)*percent)-1]))

print("min:"+str(sorted_interval[0]))
print("max:"+str(sorted_interval[int(len(sorted_interval))-1]))

average_a = np.mean(sorted_interval)
median_a = np.median(sorted_interval)
print("average:"+str(average_a))
print("median:"+str(median_a))
print("scaled_average:"+str(average_a/600))
print("scaled_median:"+str(median_a/600))
print(np.percentile(sorted_interval,10))
exit()
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# 随机生成（10000,）服从正态分布的数据
data = list(x/600 for x in sorted_interval)
"""
绘制直方图
data:必选参数，绘图数据
bins:直方图的长条形数目，可选项，默认为10
normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
facecolor:长条形的颜色
edgecolor:长条形边框的颜色
alpha:透明度
"""
plt.hist(data, bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("时间片")
# 显示纵轴标签
plt.ylabel("频率")
# 显示图标题
plt.title("频数/频率分布直方图")
plt.show()