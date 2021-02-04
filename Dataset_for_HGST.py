#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/20 19:35
@desc: 生成Train 和 Test
"""
# _*_coding:utf-8_*_
# 作者    ： YiPeiyu
# 创建时间 : 2020/11/17 10:03
import torch.nn as nn
import torch
from data_loader import Get_All_Data
import pickle
import dgl
import numpy as np
import torch.utils.data as Data
with open('./30min_in/test_x_external.file', 'rb') as fo:
    Train_x_external = pickle.load(fo)
with open('./30min_in/test_x_LSTM.file', 'rb') as fo:
    Train_x_LSTM = pickle.load(fo)
with open('./30min_in/test_x_DCNN.file', 'rb') as fo:
    Train_x_DCNN = pickle.load(fo)
with open('./30min_in/test_support_DCNN.file', 'rb') as fo:
    Train_support_DCNN = pickle.load(fo)
with open('./30min_in/test_y_LSTM.file', 'rb') as fo:
    Train_y_LSTM = pickle.load(fo)
with open('./30min_in/test_y_DCNN.file', 'rb') as fo:
    Train_y_DCNN = pickle.load(fo)
Train_x_LSTM = torch.from_numpy(Train_x_LSTM)
Train_x_external = torch.from_numpy(Train_x_external)
result = torch.cat((Train_x_LSTM, Train_x_DCNN, Train_x_external), dim=2)

a = []
for list in Train_support_DCNN:
    array =torch.stack((list[0], list[1]), dim=0)
    a.append(array)
b = torch.stack(tuple(a), dim=0).squeeze(2)

d = []
for i in range(0, 219):  # 10min：667 1627  20min：331 811  30min：219 539
    c1 = b[i][0]
    c2 = b[i][1]
    c3 = torch.cat((c1, c2), dim=1)
    d.append(c3)
e = torch.stack(tuple(d), dim=0)

f = torch.cat((result, e), dim=2)  # 训练集-x

train_y_dataset = []
Train_y_LSTM = torch.from_numpy(Train_y_LSTM)
for i in range(0, 219):  # 10min：667 1627  20min：331 811  30min：219 539
    y1 = Train_y_LSTM[i]
    y2 = Train_y_DCNN[i][:, 1]
    y = torch.stack((y1, y2), dim=1)
    train_y_dataset.append(y)
g = torch.stack(tuple(train_y_dataset), dim=0)

print(f.device)
#f = f.cuda()
print(f.device)
print(g.device)
#g = g.cuda()
print(g.device)

Dataset = Data.TensorDataset(f, g)

with open('./30min_in/Test_Dataset_for_baselines.file', 'wb') as fo:
    pickle.dump(Dataset, fo)