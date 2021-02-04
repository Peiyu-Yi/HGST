#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/19 16:50
@desc: 生成所有需要用到的数据集
"""
import pickle
import torch.utils.data as Data
import torch
import numpy as np
with open('./datasets/train_x_LSTM.file', 'rb') as fo:
    Train_x_LSTM = pickle.load(fo)
with open('./datasets/train_x_DCNN.file', 'rb') as fo:
    Train_x_DCNN = pickle.load(fo)
with open('./datasets/train_support_DCNN.file', 'rb') as fo:
    Train_support_DCNN = pickle.load(fo)
with open('./datasets/train_y_LSTM.file', 'rb') as fo:
    Train_y_LSTM = pickle.load(fo)
with open('./datasets/train_y_DCNN.file', 'rb') as fo:
    Train_y_DCNN = pickle.load(fo)
train_x_dataset = []
for i in range(0, 1627):
    x1 = Train_x_LSTM[i]
    x2 = Train_x_DCNN[i]
    x3 = Train_support_DCNN[i]
    x = [x1, x2, x3]
    train_x_dataset.append(x)
train_y_dataset = []
for i in range(0, 667):
    y1 = Train_y_LSTM[i]
    y2 = Train_y_DCNN[i]
    y = [y1, y2]
    train_y_dataset.append(y)

with open('./datasets/train_x_dataset.file', 'wb') as fo:
    pickle.dump(train_x_dataset, fo)
with open('./datasets/train_y_dataset.file', 'wb') as fo:
    pickle.dump(train_y_dataset, fo)