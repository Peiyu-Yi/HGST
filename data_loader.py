#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/15 22:11
@desc: 生成训练集，验证集，测试集
"""
import numpy as np
import csv
import torch.utils.data as Data
import pickle
import torch

#用到的站点
top_index = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,16,18,19,20,21
             ,22,23,30,31,32,33,34,36,37,38,39,46,48,51,53,54,56
              ,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,73,74,75
              ,76,77,78,81,82,83,84,87,89,90,92,93,94,95,96,97,98,99
              ,100,104,107,109]


def Get_All_Data(TG, time_lag, TG_in_one_day, forecast_day_number, TG_in_one_week, flag):
    #处理进站数据
    all_inter = []
    top_inter = []
    if flag == 1:   # 1:处理进站数据， 0：处理出站数据
        path = './data/201501_inflow_'+str(TG)+'min_data.csv'
    else:
        path = './data/201501_outflow_'+str(TG)+'min_data.csv'
    with open(path) as f:
        data = csv.reader(f, delimiter=',')
        for line in data:
            line = [int(x) for x in line]
            all_inter.append(line)
    for index in top_index:
        top_inter.append(all_inter[index])

    def get_train_data_enter(data, time_lag, TG_in_one_day, forecast_day_number, TG_in_one_week):
        data = np.array(data)
        data2 = np.zeros((data.shape[0], data.shape[1]))
        a = np.max(data)
        b = np.min(data)
        for i in range(len(data)):
            for j in range(len(data[0])):
                data2[i, j] = round((data[i, j] - b) / (a - b), 5)
        # 不包括第一周和最后一周的数据
        # not include the first week and the last week among the five weeks
        X_train_1 = [[] for i in
                     range(TG_in_one_week, len(data2[0]) - time_lag + 1 - TG_in_one_day * forecast_day_number)]
        Y_train = []
        for index in range(TG_in_one_week, len(data2[0]) - time_lag + 1 - TG_in_one_day * forecast_day_number):
            for i in range(76):
                temp = data2[i, index - TG_in_one_week: index + time_lag - 1 - TG_in_one_week].tolist()
                temp.extend(data2[i, index - TG_in_one_day: index + time_lag - 1 - TG_in_one_day])
                temp.extend(data2[i, index: index + time_lag - 1])
                X_train_1[index - TG_in_one_week].append(temp)
            Y_train.append(data2[:, index + time_lag - 1])
        X_train_1, Y_train = np.array(X_train_1), np.array(Y_train)
        print(X_train_1.shape, Y_train.shape)

        X_test_1 = [[] for i in
                    range(len(data2[0]) - TG_in_one_day * forecast_day_number, len(data2[0]) - time_lag + 1)]
        Y_test = []
        for index in range(len(data2[0]) - TG_in_one_day * forecast_day_number, len(data2[0]) - time_lag + 1):
            for i in range(76):
                temp = data2[i, index - TG_in_one_week: index + time_lag - 1 - TG_in_one_week].tolist()
                temp.extend(data2[i, index - TG_in_one_day: index + time_lag - 1 - TG_in_one_day])
                temp.extend(data2[i, index: index + time_lag - 1])
                X_test_1[index - (len(data2[0]) - TG_in_one_day * forecast_day_number)].append(temp)
            Y_test.append(data2[:, index + time_lag - 1])
        X_test_1, Y_test = np.array(X_test_1), np.array(Y_test)
        print(X_test_1.shape, Y_test.shape)

        Y_test_original = []
        for index in range(len(data[0]) - TG_in_one_day * forecast_day_number, len(data[0]) - time_lag + 1):
            Y_test_original.append(data[:, index + time_lag - 1])
        Y_test_original = np.array(Y_test_original)

        X_test_Original = [[] for i in
                    range(len(data[0]) - TG_in_one_day * forecast_day_number, len(data[0]) - time_lag + 1)]
        for index in range(len(data[0]) - TG_in_one_day * forecast_day_number, len(data[0]) - time_lag + 1):
            for i in range(76):
                temp = data[i, index - TG_in_one_week: index + time_lag - 1 - TG_in_one_week].tolist()
                temp.extend(data[i, index - TG_in_one_day: index + time_lag - 1 - TG_in_one_day])
                temp.extend(data[i, index: index + time_lag - 1])
                X_test_Original[index - (len(data[0]) - TG_in_one_day * forecast_day_number)].append(temp)
        X_test_Original = np.array(X_test_Original)

        print(Y_test_original.shape)
        print(X_test_Original.shape)

        return X_train_1, Y_train, X_test_1, Y_test, Y_test_original, X_test_Original, a, b

    X_train_1, Y_train, X_test_1, Y_test, Y_test_original, X_test_origina, a, b = get_train_data_enter(top_inter, time_lag,
                                                                                       TG_in_one_day,
                                                                                       forecast_day_number,
                                                                                       TG_in_one_week)
    return X_train_1, Y_train, X_test_1, Y_test, Y_test_original, X_test_origina, a, b


Train_in, Y_Train_in, X_Test_in, Y_Test_in, Y_Original_in, X_Original_in, a_in, b_in = Get_All_Data(TG=30,
                                                                                       time_lag=6,
                                                                                       TG_in_one_day=32,
                                                                                       forecast_day_number=7,
                                                                                       TG_in_one_week=224,
                                                                                       flag=1)
# 10min TG=10,time_lag=6,TG_in_one_day=96,forecast_day_number=7,TG_in_one_week=672
# 20min_in TG=20,time_lag=6,TG_in_one_day=48,forecast_day_number=7,TG_in_one_week=336
#30min TG=30,time_lag=6,TG_in_one_day=32,forecast_day_number=7,TG_in_one_week=224

with open('./30min_in/train_x_LSTM.file', 'wb') as fo:
    pickle.dump(Train_in, fo)
with open('./30min_in/train_y_LSTM.file', 'wb') as fo:
    pickle.dump(Y_Train_in, fo)
with open('./30min_in/test_x_LSTM.file', 'wb') as fo:
    pickle.dump(X_Test_in, fo)
with open('./30min_in/test_y_LSTM.file', 'wb') as fo:
    pickle.dump(Y_Test_in, fo)
with open('./30min_in/test_y_original_LSTM.file', 'wb') as fo:
    pickle.dump(Y_Original_in, fo)
with open('./30min_in/test_x_original_LSTM.file', 'wb') as fo:
    pickle.dump(X_Original_in, fo)

'''
10min：
x_train: 1627 * 76 * 15
y_train: 1627 * 76
x_test: 667 * 76 * 15
y_test: 667 * 76
[0:5] weekly; [5:10] daily; [10:15] recently
'''

'''
20min：
x_train: 811 * 76 * 15
y_train: 811 * 76
x_test: 331 * 76 * 15
y_test: 331 * 76
[0:5] weekly; [5:10] daily; [10:15] recently
'''