#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/25 10:31
@desc:  对比模型：LSTM网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from numpy import *
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=2)
        self.linear = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[:, 0:15]
        x = x.t().unsqueeze(-1)
        out, (h, c) = self.lstm(x)
        h = h[1]
        #h = h.squeeze(0)
        h = self.relu(h)
        logit = self.linear(h)
        # logit = F.relu(logit)
        return logit


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
# setup_seed(10)
#
#
# def train(model, Dataset, Epoch, loss_func, optimizer):
#     model.train()
#     loss_set = []
#     for epoch in range(Epoch):
#         train_losses = []
#         for step, (x, y) in enumerate(Dataset):
#             x = x.t().unsqueeze(-1)
#             out = model(x)
#             target = y.unsqueeze(-1)
#             loss = loss_func(out, target)
#             loss.backward()
#             optimizer.step()
#             train_losses.append(loss.detach().numpy())
#         loss_set.append(mean(train_losses))
#         print('Epoch [{}/{}], Loss:{:.4f}'.format(epoch+1, Epoch, mean(train_losses)))
#     plt.plot(loss_set)
#     plt.show()
#
#
# with open('../datasets/train_x_LSTM.file', 'rb') as fo:
#     x_Dataset = pickle.load(fo)
# with open('../datasets/train_y_LSTM.file', 'rb') as fo:
#     y_Dataset = pickle.load(fo)
# Dataset = Data.TensorDataset(torch.from_numpy(x_Dataset), torch.from_numpy(y_Dataset))
#
# lstm_model = LSTM(in_dim=1,
#                   hidden_dim=8)
# lstm_model.double()
# EPOCH = 100
# LR = 0.001
# optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# loss_func = nn.MSELoss()
#
# train(lstm_model, Dataset, EPOCH, loss_func, optimizer)
#
# PATH = './trained_models/LSTM_1_128_64_150_0001.pth'
# torch.save(lstm_model.state_dict(), PATH)