#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/12/4 16:54
@desc: 对比模型，全连接的神经网络 2层
"""
import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2):
        super(FNN, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        feature = x[:, 0: 33]
        h1 = self.linear1(feature)
        h1 = self.relu(h1)
        h2 = self.linear2(h1)
        h2 = self.relu(h2)
        Out = self.linear3(h2)
        return Out