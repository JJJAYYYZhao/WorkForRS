#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time

import torch

from utils import build_graph, Data, split_validation
from model import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=5, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
# parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')

# 新添参数
parser.add_argument("--hidden_dropout_prob", default=0.2, type=float)
# seq的统一长度控制
parser.add_argument("--max_seq_length", type=int)
# timenoise的权重控制参数（已弃用）
parser.add_argument("--a", default=0.3, type=float)#[0.1, 0.2, 0.3, 0.4]
# 对时间片的缩放控制参数
parser.add_argument("--time_scale",type=int)
# 对时间片的上限控制参数
parser.add_argument("--time_max",type=int)
parser.add_argument("--time_max_percent",default=0.5,type=int)#[0.4,0.5,0.6]
# 对时间差的上限控制参数
parser.add_argument("--interval_limit",type=int)
parser.add_argument("--interval_limit_percent",default=0.05,type=int)#[0.04,0.05,0.06]
# 单独进行测试集的测试
parser.add_argument("--test_or_train",default="train")

opt = parser.parse_args()
# 避免结果覆写记录
version=1
result_dir='results/'+opt.dataset+'_batchsize-'+str(opt.batchSize)+'_lr-'+str(opt.lr)+'_lrdc-'+str(opt.lr_dc)+'_hidden-'+str(opt.hiddenSize)+'_l2-'+str(opt.l2)+'_v-'
while(True):
    if os.path.exists(result_dir+str(version)):
        version+=1
        continue
    os.makedirs(result_dir+str(version))
    break
writer = SummaryWriter(log_dir=result_dir+str(version))
# 超参数记录
writer.add_text(tag="Parameters", text_string=str(opt))
def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    # 确定时间片、时间差上限参数
    total_data = pickle.load(open('../datasets/' + opt.dataset + '/total.txt', 'rb'))
    interval=total_data[2]
    time_slice = sorted(list(
        int(reduce(lambda x, y: x + y, interval[i][j:])) for i in range(len(interval)) for j in
        range(len(interval[i]))))
    total_interval = []
    for item in interval:
        total_interval += item
    time_interval = sorted(total_interval)
    if opt.dataset == 'diginetica':
        n_node = 43098
        opt.time_scale=600
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
        opt.time_scale = 1
    else:
        n_node = 310
        opt.time_scale = 600
    opt.time_max = int(np.percentile(time_slice, opt.time_max_percent * 100)/opt.time_scale)
    opt.interval_limit = int(np.percentile(time_interval, opt.interval_limit_percent * 100))
    # 确定max_seq_length参数
    max_seq_length = max([len(seq) for seq in total_data[0]])
    opt.max_seq_length=(max_seq_length//10+1)*10
    print(opt)
    model = trans_to_cuda(SessionGraph(opt, n_node))
    if opt.test_or_train=='train':
        # train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
        train_data = Data(train_data, opt, shuffle=True)
        test_data = Data(test_data, opt, shuffle=False)

        start = time.time()
        best_result = [0, 0]
        best_epoch = [0, 0]
        bad_counter = 0
        for epoch in range(opt.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            hit, mrr, total_loss = train_test(model, train_data, test_data)
            flag = 0
            if hit >= best_result[0]:
                best_result[0] = hit
                best_epoch[0] = epoch
                flag = 1
            if mrr >= best_result[1]:
                best_result[1] = mrr
                best_epoch[1] = epoch
                flag = 1
            # 训练记录
            writer.add_scalar(tag="Record/Loss",
                              scalar_value=total_loss,
                              global_step=epoch
                              )
            writer.add_scalar(tag="Record/Recall",
                              scalar_value=hit,
                              global_step=epoch
                              )
            writer.add_scalar(tag="Record/MMR",
                              scalar_value=mrr,
                              global_step=epoch
                              )
            # 最好结果记录
            writer.add_scalar(tag="Best_Result/Recall",
                              scalar_value=best_result[0],
                              global_step=best_epoch[0]
                              )
            writer.add_scalar(tag="Best_Result/MRR",
                              scalar_value=best_result[1],
                              global_step=best_epoch[1]
                              )
            print('Current Result:')
            print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\t' % (hit, mrr))
            print('Best Result:')
            print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
            bad_counter += 1 - flag
            if bad_counter >= opt.patience:
                break
        print('-------------------------------------------------------')
        end = time.time()
        print("Run time: %f s" % (end - start))
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(model.state_dict(),'model/'+opt.dataset+'_'+str(opt.epoch)+'_a='+str(opt.a)+'_seqlen='+str(opt.max_seq_length)+'.pth')
        writer.close()
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
        test_data = Data(test_data, opt, shuffle=False)
        model.load_state_dict(torch.load('model/sample_5_a=0.3_seqlen=20.pth'))
        start = time.time()
        print('start predicting: ', datetime.datetime.now())
        model.eval()
        hit, mrr = [], []
        slices = test_data.generate_batch(model.batch_size)
        for i in slices:
            targets, scores = forward(model, i, test_data)
            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                hit.append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        print('-------------------------------------------------------')
        end = time.time()
        print("Run time: %f s" % (end - start))
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit,mrr))





if __name__ == '__main__':
    main()
