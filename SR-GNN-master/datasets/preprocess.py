#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset == 'yoochoose':
    dataset = 'yoochoose-clicks.dat'
    # dataset = 'sample.dat'

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',',fieldnames=['session_id','timestamp','item_id','category'])
    else:
        reader = csv.DictReader(f, delimiter=';',fieldnames=['session_id','user_id','item_id','timeframe','eventdate'])
    sess_clicks = {}
    # sess_timestamps记录每段对话的点击时间，后续处理sess_clicks时也相应的处理sess_timestamps
    sess_timestamps = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    head_row=next(reader)
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id'], time.mktime(time.strptime(data['timestamp'][:19], '%Y-%m-%dT%H:%M:%S'))
        else:
            item = data['item_id'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_timestamps[i] = [c[1] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]
        del sess_timestamps[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

# 筛选出点击次数大于5次的item序列
length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    # 对时间戳序列同样做筛选操作
    # ======start======
    curseq_time = sess_timestamps[s]
    filseq_time = [item for index, item in enumerate(curseq_time) if iid_counts[curseq[index]] >= 5]
    # ======end======
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
        del sess_timestamps[s]
    else:
        sess_clicks[s] = filseq
        sess_timestamps[s] = filseq_time

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)  # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
print(len(tra_sess))  # 186670    # 7966257
print(len(tes_sess))  # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}


# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    train_timestamps = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        seq_time = sess_timestamps[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
        train_timestamps += [seq_time]
    print(item_ctr)  # 43098, 37484
    return train_ids, train_dates, train_seqs, train_timestamps


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    test_timestamps = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        seq_time = sess_timestamps[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
        test_timestamps += [seq_time]
    return test_ids, test_dates, test_seqs, test_timestamps


# 新添加返回参数：序列的时间戳tra_timestamps、tes_timestamps，在函数obtain_tra中已有对应的修改
tra_ids, tra_dates, tra_seqs, tra_timestamps = obtian_tra()
tes_ids, tes_dates, tes_seqs, tes_timestamps = obtian_tes()


def process_seqs(iseqs, idates, itimestamps):
    out_seqs = []
    out_dates = []
    out_time_interval = []
    labs = []
    ids = []
    for id, seq, date, timestamps in zip(range(len(iseqs)), iseqs, idates, itimestamps):
        time_interval = [timestamps[i + 1] - timestamps[i] for i in range(0, len(timestamps) - 1)]

        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_time_interval += [time_interval[:len(seq[:-i])]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, out_time_interval, labs, ids


# 新添加返回参数：各点击的时间间隔序列tr_time_invertal、te_time_invertal，在函数process_seqs中已有对应的修改
tr_seqs, tr_dates, tr_time_invertal, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates, tra_timestamps)
te_seqs, te_dates, te_time_interval, te_labs, te_ids = process_seqs(tes_seqs, tes_dates, tes_timestamps)
tra = (tr_seqs, tr_labs, tr_time_invertal)
tes = (te_seqs, te_labs, te_time_interval)
total = (tr_seqs+te_seqs, tr_labs+te_labs, tr_time_invertal+te_time_interval)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all / (len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(total, open('diginetica/total.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:],tr_time_invertal[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:],tr_time_invertal[-split64:])
    total4, total64 = (tr_seqs[-split4:]+te_seqs, tr_labs[-split4:]+te_labs,tr_time_invertal[-split4:]+te_time_interval), (tr_seqs[-split64:]+te_seqs, tr_labs[-split64:]+te_labs,tr_time_invertal[-split64:]+te_time_interval)
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(total4, open('yoochoose1_4/total.txt', 'wb'))
    pickle.dump(total64, open('yoochoose1_64/total.txt', 'wb'))

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))
else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
