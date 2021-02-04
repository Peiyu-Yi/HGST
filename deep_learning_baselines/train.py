#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/12/5 13:23
@desc:
"""
import pickle
import torch
import torch.nn as nn
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from FNN import FNN
from GCN import GCN
from LSTM import LSTM

# step 1  加载训练数据集
#torch.load(map_location=torch.device('cpu'))
with open('../30min_out/Train_Dataset_for_baselines.file', 'rb') as fo:
    Dataset = pickle.load(fo)

# step2 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
setup_seed(10)

# step3 定义训练过程
def Train(model, Dataset, Epoch, loss_func, optimizer, name_str):
    model.train()
    loss_set = []
    for epoch in range(Epoch):
        train_loss = []
        for step, (x, y) in enumerate(Dataset):
            #if name_str == 'GCN':
            #   x = x.float()
            x = x.float().cuda()
            out = model(x)
            target = y[:, 1].unsqueeze(1).float().cuda()  # 0:in 1:out
            loss = loss_func(out, target)
            optimizer.zero_grad()
            #model.cleargrad()
            #loss = loss.float()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().detach().numpy())
        loss_set.append(mean(train_loss))
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, Epoch, mean(train_loss)))
    plt.plot(loss_set)
    plt.show()
    PATH = './' + name_str + '.pth'
    torch.save(model.state_dict(), PATH)


Method = 'LSTM'  # FNN, LSTM, GCN

if Method == 'FNN':
    model = FNN(in_dim=33, hidden_dim1=64, hidden_dim2=32)
    #model.double()
    EPOCH = 100
    LR = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    loss_func = nn.MSELoss()
    Train(model, Dataset, EPOCH, loss_func, optimizer, 'FNN_30min_out')

if Method == 'LSTM':
    model = LSTM(in_dim=1, hidden_dim=32)
    model = model.cuda()
    #model.double()
    EPOCH = 50
    LR = 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    loss_func = nn.MSELoss()
    Train(model, Dataset, EPOCH, loss_func, optimizer, 'LSTM_30min_out')

if Method == 'GCN':
    model = GCN(in_dim=10, hidden_dim=64, out_dim=1)
    #model.double()
    EPOCH = 100
    LR = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    loss_func = nn.MSELoss()
    Train(model, Dataset, EPOCH, loss_func, optimizer, 'GCN_30min_out')

