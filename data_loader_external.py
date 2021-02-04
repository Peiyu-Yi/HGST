#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/12/1 10:45
@desc: 生成external的数据集  训练集：1627*76*8  1627*76*1  测试集：667*76*8 667*76*1
"""
import pickle
import time
import datetime
import numpy as np
import pytz
from numpy import *
import pandas as pd
from chinese_calendar import is_workday, is_holiday

f = pd.read_csv('./data/weather.csv')
# 对天气数据min-max归一化处理
quant_features = ['T', 'U', 'Ff', 'VV', 'Td', 'RRR']
scaled_features = {}
for each in quant_features:
    min, max = f[each].min(), f[each].max()
    scaled_features[each] = [min, max]
    f.loc[:, each] = (f[each] - min) / (max - min)
#f = f.iloc[::-1]
#f = f.reset_index(drop=True)
f['time'] = pd.to_datetime(f['time'], format='%d.%m.%Y %H:%M')
f.set_index('time', inplace=True)

#10min: 训练集：1月8日7：20 -1月24日22：30  测试集：1月25日7：20 -1月31日22：30
#20min：训练集: 1月8日8:10-1月24日22：30  测试集：1月25日8：10-1月31日22：30
#30min: 训练集: 1月8日9:00-1月24日22：30  测试集:1月25日9：00-1月31日22：30
train_start_time = datetime.datetime(2015, 1, 8, 9, 00)
train_stop_time = datetime.datetime(2015, 1, 24, 22, 30)
test_start_time = datetime.datetime(2015, 1, 25, 9,00)
test_stop_time = datetime.datetime(2015, 1, 31, 22, 30)

#time_slots = []
#time_slots_between = []
#index_to_delete = [91]


def is_time_between(hour, minute):
    s = pytz.utc.localize(datetime.time(6, 30))
    e = pytz.utc.localize(datetime.time(22, 30))
    cur = pytz.utc.localize(datetime.time(hour, minute))
    return s <= cur <= e


def Get_external_dataset(start_time, stop_time):
    time_slots = []
    time_slots_between = []
    index_to_delete = [27]  # 10min：91  20min：43 30min：27
    num_days = (stop_time - start_time).days
    while start_time <= stop_time:
        time_slots.append(start_time)
        start_time = start_time + datetime.timedelta(minutes=30)  # 10min 20min 30min
    for time in time_slots:
        hour = time.hour
        minute = time.minute
        if is_time_between(hour, minute):
            time_slots_between.append(time)
    for i in range(1, num_days + 1):
        index = 27 + i*33 #10min：91 97  20min：43 49  30min：27  33
        index_to_delete.append(index)
    time_slots_between = [time_slots_between[i] for i in range(0, len(time_slots_between), 1) if i not in index_to_delete]

    external_feature = []
    for time in time_slots_between:
        day = time.day
        cut_time = '2015-1-'+str(day)
        temp_f = f[cut_time]
        diff = temp_f.index - time
        temp_f['diff'] = abs(diff)
        idx = temp_f['diff'].idxmin()
        row = temp_f.loc[idx]   # 该时段的天气特征（数值 已min-max归一化） 6维 + 时间差（不用）

        dim_0 = int(is_workday(time))
        dim_1 = int(is_holiday(time))  # 该时间段的节假日特征（one-hot编码） 2维 [是否工作日，是否节假日]

        feature = np.array((dim_0, dim_1, row[0], row[1], row[2], row[3], row[4], row[5]))
        feature = tile(feature, (76, 1))  # 76个站点， 同一个时间external feature一样
        external_feature.append(feature)
    external_feature = np.array(external_feature)
    return external_feature


train_external_dataset_x = Get_external_dataset(train_start_time, train_stop_time)

test_external_dataset_x = Get_external_dataset(test_start_time, test_stop_time)

with open('./30min_in/train_x_external.file', 'wb') as fo:
    pickle.dump(train_external_dataset_x, fo)
with open('./30min_in/test_x_external.file', 'wb') as fo:
    pickle.dump(test_external_dataset_x, fo)
#external_feature = np.expand_dims(external_feature, axis=1)

