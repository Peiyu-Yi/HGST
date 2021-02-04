#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/12/1 21:22
@desc: FNN网络处理external factors
"""
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2):
        super(FNN, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        h = self.linear1(x)
        h = F.relu(h)
        h = self.linear2(h)
        h = F.relu(h)
        out = self.linear3(h)
        return out
