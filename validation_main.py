#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/23 20:25
@desc: 验证部分
"""
from numpy import *
import torch
import torch.nn as nn
import pickle
import math
import numpy as np
import numpy.linalg as la
from sklearn.metrics import mean_squared_error, mean_absolute_error
from HGST import HGST

MAX_VALUE = 1639
MIN_VALUE = 0
with open('./datasets/Test_Dataset_for_HGST.file', 'rb') as fo:
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
        print(x.device)
        print(y.device)
        output = model(x)
        target_cuda = y[:, 0].unsqueeze(1)
        loss = loss_func(output, target_cuda)
        rmse, mae, acc, r2_score, var_score = evaluation(output.cpu().detach().numpy(), target_cuda.cpu().detach().numpy())
        test_loss.append(loss.cpu().detach().numpy())
        test_rmse.append(rmse * MAX_VALUE)
        test_mae.append(mae * MAX_VALUE)
        test_acc.append(acc)
        test_r2.append(r2_score)
        test_var.append(var_score)
    print('HGST_rmse:%r' % (np.mean(test_rmse)),
          'HGST_mae:%r' % (np.mean(test_mae)),
          'HGST_acc:%r' % (np.mean(test_acc)),
          'HGST_r2:%r' % (np.mean(test_r2)),
          'HGST_var:%r' % (np.mean(test_var)))


model = HGST(lstm_in_dim=1,
             lstm_out_dim=16,
             dcnn_in_dim=10,
             dcnn_out_dim=2,
             dcnn_layers=2)
load_PATH = './trained_models/HGST_1_16_10_2_2_100_0000001.pth'
model.load_state_dict(torch.load(load_PATH))
model.double()
model_cuda = model.cuda()
loss_func = nn.MSELoss()
Test(model_cuda, Test_Dataset, loss_func)