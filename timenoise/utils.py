import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    # 在[l，r]之间找一个不在s中的随机数
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    time_noise_figure = np.array([time_span] * size)
    for i in range(size):
        for j in range(size):
            span_demo = time_seq[i] - time_seq[j]

            span = abs(span_demo)
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, UsersItem_diff, result_queue,
                    SEED):
    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]
        time_diff = np.zeros([maxlen], dtype=np.int32)

        idx = maxlen - 1
        ts = set(map(lambda x: x[0], user_train[user]))
        # print(UsersItem_diff)
        for i in reversed(user_train[user][:-1]):
            # time_diff[idx] = diff[idx]
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt  # 记录了顺序（正向的，positive）的item序号的sequence
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        diff = UsersItem_diff[user][:-1]
        for i in reversed(diff):
            # print(i)
            time_diff[idx] = i
            if idx == -1: break

        time_matrix = relation_matrix[user]
        # print(time_seq)
        # print()
        return (user, seq, time_seq, time_matrix, pos, neg, time_diff)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, UsersItem_diff, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
                                                      UsersItem_diff,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:  # float as map key?
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map, args):
    User_filted = dict()
    user_set = set()
    item_set = set()
    # 重新给user和item赋予id（从1开始）
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])  # 按照时间排序，默认从小到大

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))

    time_max = set()
    UsersItem_diff = dict()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = []
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                if time_list[i + 1] - time_list[i] < args.interval_max:
                    time_diff.append(time_list[i + 1] - time_list[i])
                else:
                    time_diff.append(args.interval_max)
        UsersItem_diff[user] = time_diff
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max), UsersItem_diff


def data_partition(fname, args):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open('data/%s.txt' % fname, 'r')  # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:  # hard-coded
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum, UsersItem_diff = cleanAndsort(User, time_map, args)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, timenum, UsersItem_diff]


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum, UsersItem_diff] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        time_diff = np.zeros([args.maxlen], dtype=np.int32)

        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        diff = UsersItem_diff[u][:-1]
        idx = args.maxlen - 1
        for i in reversed(diff):
            # print(i)
            time_diff[idx] = i
            if idx == -1: break
        time_diff = time_diff[np.newaxis,:]
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx, time_diff]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum, UsersItem_diff] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_diff = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        diff = UsersItem_diff[u][:-1]
        idx = args.maxlen - 1
        for i in reversed(diff):
            # print(i)
            time_diff[idx] = i
            if idx == -1: break
        time_diff = time_diff[np.newaxis, :]
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx, time_diff]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
