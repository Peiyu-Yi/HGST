#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/18 11:56
@desc: dcnn的训练集，测试集
"""
from data_loader import Get_All_Data
import torch
import pickle
import datetime
import time
import pytz
import numpy as np
import dgl
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import torch.utils.data as Data


def Get_DCNN_data():
    #进站数据
    X_Train_in, Y_Train_in, X_Test_in, Y_Test_in, Y_Original_in, X_Original_in, a_in, b_in = Get_All_Data(TG=30,
                                                                                       time_lag=6,
                                                                                       TG_in_one_day=32,
                                                                                       forecast_day_number=7,
                                                                                       TG_in_one_week=224,
                                                                                       flag=1)
    #出站数据
    X_Train_out, Y_Train_out, X_Test_out, Y_Test_out, Y_Original_out, X_Original_out, a_out, b_out = Get_All_Data(TG=30,
                                                                                       time_lag=6,
                                                                                       TG_in_one_day=32,
                                                                                       forecast_day_number=7,
                                                                                       TG_in_one_week=224,
                                                                                       flag=0)
    # 1627 * 76 * 10
    train_feature = torch.cat((torch.from_numpy(X_Train_in[:, :, 10:15]),
                           torch.from_numpy(X_Train_out[:, :, 10:15])), dim=2)
    # 1627 * 76 * 2
    train_y = torch.stack((torch.from_numpy(Y_Train_in),
                    torch.from_numpy(Y_Train_out)), dim=2)
    # 667 * 76 *  10
    test_feature = torch.cat((torch.from_numpy(X_Test_in[:, :, 10:15]),
                           torch.from_numpy(X_Test_out[:, :, 10:15])), dim=2)
    # 667 *  76 * 2
    test_y = torch.stack((torch.from_numpy(Y_Test_in),
                    torch.from_numpy(Y_Test_out)), dim=2)
    return train_feature, train_y, test_feature, test_y


weekdays_morning_rush_hours = ['6', '7', '8', '9']
weekdays_day_hours = ['10', '11', '12', '13', '14', '15']
weekdays_evening_rush_hours = ['16', '17', '18']
weekdays_night_hours = ['19', '20', '21', '22']

weekends_morning_hours = ['6', '7', '8', '9']
weekends_trip_hours = ['10', '11', '12', '13', '14', '15', '16', '17', '18']
weekends_evening_hours = ['19', '20', '21', '22']

weekdays = ['8', '9', '12', '13', '14', '15', '16', '19', '20', '21', '22', '23', '26', '27', '28', '29', '30']
weekends = ['10', '11', '17', '18', '24', '25', '31']


def day_kind_check(d, s):
    dkind = ''
    if d in weekdays:
        if s in weekdays_morning_rush_hours:
            dkind = 'morning_rush_hour_on_weekdays'
        if s in weekdays_day_hours:
            dkind = 'day_hour_on_weekdays'
        if s in weekdays_evening_rush_hours:
            dkind = 'evening_rush_hour_on_weekdays'
        if s in weekdays_night_hours:
            dkind = 'night_hour_on_weekdays'
    else:
        if s in weekends_morning_hours:
            dkind = 'morning_hour_on_weekends'
        if s in weekends_trip_hours:
            dkind = 'trip_hour_on_weekends'
        if s in weekends_evening_hours:
            dkind = 'evening_hour_on_weekends'
    return dkind


with open('./data/metro_hin_top.file', 'rb') as fo:
    metro_hin = pickle.load(fo)


def is_time_between(hour, minute):
    s = pytz.utc.localize(datetime.time(6, 30))
    e = pytz.utc.localize(datetime.time(22, 30))
    cur = pytz.utc.localize(datetime.time(hour, minute))
    return s <= cur <= e


def Get_Edge_types(start_time, stop_time):
    init_start_time = start_time
    edge_types = []
    time_slots = []
    time_slots_between = []
    index_to_delete = [27]  # 10min:91  20min:43 30min:27
    while start_time <= stop_time:
        time_slots.append(start_time)
        start_time = start_time +datetime.timedelta(minutes=30) # 10min 20min 30min
    for time in time_slots:
        hour = time.hour
        minute = time.minute
        if is_time_between(hour, minute):
            time_slots_between.append(time)
    for time in time_slots_between:
        day = time.day
        hour = time.hour
        daykind = day_kind_check(str(day), str(hour))
        edge_types.append(daykind)
    num_days = (stop_time - init_start_time).days
    for i in range(1, num_days+1):
        index = 27 + i * 33  # 10min:91 97  20min:43 49 30min:27 33
        index_to_delete.append(index)
    edge_types = [edge_types[i] for i in range(0, len(edge_types), 1) if i not in index_to_delete]
    return edge_types

#10min: 训练集：1月8日7：20 -1月24日22：30  测试集：1月25日7：20 -1月31日22：30
#20min：训练集: 1月8日8:10-1月24日22：30  测试集：1月25日8：10-1月31日22：30
#30min: 训练集: 1月8日9:00-1月24日22：30  测试集:1月25日9：00-1月31日22：30


train_start_time = datetime.datetime(2015, 1, 8, 9, 00)
train_stop_time = datetime.datetime(2015, 1, 24, 22, 30)
test_start_time = datetime.datetime(2015, 1, 25, 9, 00)
test_stop_time = datetime.datetime(2015, 1, 31, 22, 30)

# 1627个训练样本的边类型 1*1627
train_edge_types = Get_Edge_types(train_start_time, train_stop_time)
# 667个测试样本的边类型 1*667
test_edge_types = Get_Edge_types(test_start_time, test_stop_time)


def calculate_random_walk_matrix(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    d = np.array(adj_matrix.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_matrix).astype(np.float32).todense()
    return random_walk_mx

s_num = 76


def Get_Support_matrix():
    train_support = []
    test_support = []
    for dkind in train_edge_types:
        support_1 = torch.zeros((1, s_num, s_num))
        support_2 = torch.zeros((1, s_num, s_num))
        s_1 = torch.zeros((1, s_num))
        s_2 = torch.zeros((1, s_num))
        support = []

        sub_g = metro_hin.edge_type_subgraph([dkind])
        adj_mat = sub_g.adjacency_matrix().to_dense()
        s_1 = torch.cat((s_1, torch.tensor(calculate_random_walk_matrix(adj_mat).T)), 0)
        s_2 = torch.cat((s_2, torch.tensor(calculate_random_walk_matrix(adj_mat.T).T)), 0)

        ssize = s_1.shape[0]
        s_1 = s_1[1:ssize, :].reshape((-1, s_num, s_num))
        s_2 = s_2[1:ssize, :].reshape((-1, s_num, s_num))

        support_1 = torch.cat((support_1, s_1), 0)
        support_2 = torch.cat((support_2, s_2), 0)

        ssize = support_1.shape[0]
        support_1 = support_1[1:ssize].reshape((-1, s_num, s_num))
        support_2 = support_2[1:ssize].reshape((-1, s_num, s_num))

        support.append(support_1)
        support.append(support_2)
        train_support.append(support)
    for dkind in test_edge_types:
        support_1 = torch.zeros((1, s_num, s_num))
        support_2 = torch.zeros((1, s_num, s_num))
        s_1 = torch.zeros((1, s_num))
        s_2 = torch.zeros((1, s_num))
        support = []

        sub_g = metro_hin.edge_type_subgraph([dkind])
        adj_mat = sub_g.adjacency_matrix().to_dense()
        s_1 = torch.cat((s_1, torch.tensor(calculate_random_walk_matrix(adj_mat).T)), 0)
        s_2 = torch.cat((s_2, torch.tensor(calculate_random_walk_matrix(adj_mat.T).T)), 0)

        ssize = s_1.shape[0]
        s_1 = s_1[1:ssize, :].reshape((-1, s_num, s_num))
        s_2 = s_2[1:ssize, :].reshape((-1, s_num, s_num))

        support_1 = torch.cat((support_1, s_1), 0)
        support_2 = torch.cat((support_2, s_2), 0)

        ssize = support_1.shape[0]
        support_1 = support_1[1:ssize].reshape((-1, s_num, s_num))
        support_2 = support_2[1:ssize].reshape((-1, s_num, s_num))

        support.append(support_1)
        support.append(support_2)
        test_support.append(support)
    return train_support, test_support


# 1627* [1*76*76]
#train_support_matrix = Get_Support_matrix(train_edge_types)

# 667 * [1*76*76]
#test_support_matrix = Get_Support_matrix(test_edge_types)

# 训练集 1627*76*10, 1627*76*2,   测试集 667*76*10, 667*76*2
train_x, train_y, test_x, test_y = Get_DCNN_data()
# 训练集 1627*2*[1*76*76]  测试集 667*2*[1*76*76]
train_support, test_support = Get_Support_matrix()

# Train_dataset = Data.TensorDataset(train_x, train_y)
# Test_dataset = Data.TensorDataset(test_x, test_y)
#
# with open('./data/Train_DCNN_dataset.file', 'wb') as fo:
#     pickle.dump(Train_dataset, fo)
# with open('./data/Test_DCNN_dataset.file', 'wb') as fo:
#     pickle.dump(Test_dataset, fo)
# with open('./data/Train_DCNN_support.file', 'wb') as fo:
#     pickle.dump(train_support, fo)
# with open('./data/Test_DCNN_support.file', 'wb') as fo:
#     pickle.dump(test_support, fo)
with open('./30min_in/train_x_DCNN.file', 'wb') as fo:
    pickle.dump(train_x, fo)
with open('./30min_in/train_y_DCNN.file', 'wb') as fo:
    pickle.dump(train_y, fo)
with open('./30min_in/test_x_DCNN.file', 'wb') as fo:
    pickle.dump(test_x, fo)
with open('./30min_in/test_y_DCNN.file', 'wb') as fo:
    pickle.dump(test_y, fo)
with open('./30min_in/train_support_DCNN.file', 'wb') as fo:
    pickle.dump(train_support, fo)
with open('./30min_in/test_support_DCNN.file', 'wb') as fo:
    pickle.dump(test_support, fo)
