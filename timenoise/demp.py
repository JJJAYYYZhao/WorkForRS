import torch
import numpy as np

# a = np.random.rand(3, 2)
# print(a)
# a = a[:,np.newaxis,:]
# print(a.shape)
#
# # c = a[-50:]
# c = np.tile(a, (1, 2, 1))
# print(c)
# print(c.shape)
# print(c.shape)
# print(a.shape)
# a =np.zeros([5])
# print(a.shape)
# a = np.zeros(5)
# print(a.shape)
import os
import time
import torch

# torch.cuda.set_device(3)
import pickle
import argparse

from model import TiSASRec
from tqdm import tqdm
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=601, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--interval_max', default=16, type=int)

args = parser.parse_args()
dataset = data_partition(args.dataset, args)
[user_train, user_valid, user_test, usernum, itemnum, timenum, UsersItem_diff] = dataset

time = []
all_num = 0
num256 = 0
for u, time_diff in UsersItem_diff.items():
    num = len(time_diff)
    # print(time_diff)
    # print(num)
    time.extend(time_diff)
    # print(time)
    all_num = all_num + num

print(all_num)
final_time = sorted(time)
# print(len(time))
mean = sum(final_time)/all_num
print(mean)
print(max(time))
# print(min(time))
print(final_time[230000])
# print()


