#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/23 15:10
@desc: HA ARIMA SVR 模型
"""
import pandas as pd
import pickle
import numpy as np
import numpy.linalg as la
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.tsa.arima_model import ARIMA
import csv
from sklearn.svm import SVR

# 10min:in_1639 out_2451
# 20min: in_2946 out_4292
# 30min: in_4326 out_6328
MAX_VALUE = 2451
MIN_VALUE = 0


def evaluate(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b) / la.norm(a)
    r2 = 1 - ((a - b)**2).sum() / ((a - a.mean())**2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1-F_norm, r2, var


top_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21
             , 22, 23, 30, 31, 32, 33, 34, 36, 37, 38, 39, 46, 48, 51, 53, 54, 56
              , 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75
              , 76, 77, 78, 81, 82, 83, 84, 87, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99
              , 100, 104, 107, 109]

path = '../data/201501_outflow_10min_data.csv'
all_inter = []
top_enter = []
with open(path) as f:
    data = csv.reader(f, delimiter=',')
    for line in data:
        line = [int(x) for x in line]
        all_inter.append(line)
for index in top_index:
    top_enter.append(all_inter[index])
# 76个站点的历史数据 归一化的数据
data = pd.DataFrame(top_enter).T

# 667个测试样本 没有归一化的数据
with open('../10min_out/test_x_original_LSTM.file', 'rb') as fo:
    testX = pickle.load(fo)
with open('../10min_out/test_y_original_LSTM.file', 'rb') as fo:
    testY = pickle.load(fo)

# 1627个训练样本 归一化后 用于lstm 1627*76*15
with open('../10min_out/train_x_LSTM.file', 'rb')  as fo:
    train_lstm_x = pickle.load(fo)
with open('../10min_out/train_y_LSTM.file', 'rb') as fo:
    train_lstm_y = pickle.load(fo)
# 667个测试样本 归一化后 同于lstm 667*76*1
with open('../10min_out/test_x_LSTM.file', 'rb') as fo:
    test_lstm_x = pickle.load(fo)
with open('../10min_out/test_y_LSTM.file', 'rb') as fo:
    test_lstm_y = pickle.load(fo)


method = 'SVR'   # HA ARIMA SVR

if method == 'SVR':
    rmse, mae, acc, r2, var = [], [], [], [], []
    for i in range(76):
        train_x_data = train_lstm_x[:, i, :]
        train_y_data = train_lstm_y[:, i]
        test_x_data = test_lstm_x[:, i, :]
        test_y_data = test_lstm_y[:, i]

        svr_model = SVR(kernel='linear',)
        svr_model.fit(train_x_data, train_y_data)
        pre = svr_model.predict(test_x_data)
        irmse, imae, iacc, ir2, ivar = evaluate(test_y_data, pre)
        rmse.append(irmse * MAX_VALUE)
        mae.append(imae * MAX_VALUE)
        acc.append(iacc)
        r2.append(ir2)
        var.append(ivar)
    acc = np.mat(acc)
    acc[acc < 0] = 0
    print('SVR_rmse:%r'%(np.mean(rmse)),
          'SVR_mae:%r'%(np.mean(mae)),
          'SVR_acc:%r'%(np.mean(acc)),
          'SVR_r2:%r'%(np.mean(r2)),
          'SVR_var:%r'%(np.mean(var)))


if method == 'ARIMA':
    date_time_index = []
    for day in range(1, 32):
        start = str(day) + '/1/2015 6:30:00'
        end = str(day) + '/1/2015 22:20:00'
        times = pd.date_range(start=start, end=end, freq='10min') # 10 20 30
        for item in times:
            date_time_index.append(item)
    #rng = pd.date_range(start='1/1/2015 6:30:00', end='1/31/2015 22:30:00', freq='10min')
    a1 = pd.DatetimeIndex(date_time_index)
    data.index = a1
    num = data.shape[1]
    rmse, mae, acc, r2, var, pred, ori = [], [], [], [], [], [], []
    for i in range(num):
        ts = data.iloc[:, i]
        ts_log = np.log(ts)
        ts_log = np.array(ts_log, dtype=np.float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        ts_log = pd.Series(ts_log)
        ts_log.index = a1
        model = ARIMA(ts_log, order=[1, 0, 0])
        properModel = model.fit()
        predict_ts = properModel.predict(6, dynamic=True)
        log_recover = np.exp(predict_ts)
        ts = ts[log_recover.index]
        er_rmse, er_mae, er_acc, r2_score, var_score = evaluate(ts, log_recover)
        rmse.append(er_rmse)
        mae.append(er_mae)
        acc.append(er_acc)
        r2.append(r2_score)
        var.append(var_score)
    acc1 = np.mat(acc)
    acc1[acc1 < 0] = 0
    print('arima_rmse:%r'%(np.mean(rmse)),
          'arima_mae:%r'%(np.mean(mae)),
          'arima_acc:%r'%(np.mean(acc1)),
          'arima_r2:%r'%(np.mean(r2)),
          'arima_var:%r'%(np.mean(var)))


if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = testX[i]
        a1 = a[:, 0:5]
        a2 = np.mean(a1, axis=1)
        result.append(a2)
    result1 = np.array(result)
    rmse, mae, accuracy, r2, var = evaluate(testY, result1)
    print('HA_rmse:%r'%rmse,
          'HA_mae:%r'%mae,
          'HA_acc:%r'%accuracy,
          'HA_r2:%r'%r2,
          'HA_var:%r'%var)