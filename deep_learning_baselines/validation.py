#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/12/5 15:00
@desc:
"""
from numpy import *
import torch
import torch.nn as nn
import pickle
import math
import numpy as np
import numpy.linalg as la
from sklearn.metrics import mean_squared_error, mean_absolute_error
from FNN import FNN
from LSTM import LSTM
from GCN import GCN

# 10min:in_1639 out_2451
# 20min: in_2946 out_4292
# 30min: in_4326 out_6328
MAX_VALUE = 6328
MIN_VALUE = 0
with open('../30min_out/Test_Dataset_for_baselines.file', 'rb') as fo:
    Test_Dataset = pickle.load(fo)


def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    r2 = 1 - ((a - b)**2).sum() / ((a - a.mean())**2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1-F_norm, r2, var


def Test(model, Test_Dataset, loss_func):
    model.eval()
    test_loss, test_rmse, test_mae, test_acc, test_r2, test_var = [], [], [], [], [], []
    for step, (x, y) in enumerate(Test_Dataset):
        x = x.float()
        output = model(x)
        target = y[:, 1].unsqueeze(1).float()  # 0:in  1:out
        loss = loss_func(output, target)
        rmse, mae, acc, r2_score, var_score = evaluation(output.detach().numpy(), target.detach().numpy())
        test_loss.append(loss.detach().numpy())
        test_rmse.append(rmse * MAX_VALUE)
        test_mae.append(mae * MAX_VALUE)
        test_acc.append(acc)
        test_r2.append(r2_score)
        test_var.append(var_score)
    print('rmse:%r' % (np.mean(test_rmse)),
          'mae:%r' % (np.mean(test_mae)),
          'acc:%r' % (np.mean(test_acc)),
          'r2:%r' % (np.mean(test_r2)),
          'var:%r' % (np.mean(test_var)))


model = LSTM(in_dim=1, hidden_dim=32)
load_PATH = './LSTM_30min_out.pth'
model.load_state_dict(torch.load(load_PATH))
#model.double()
#model_cuda = model.cuda()
loss_func = nn.MSELoss()
Test(model, Test_Dataset, loss_func)