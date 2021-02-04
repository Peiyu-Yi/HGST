#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/21 20:50
@desc: 训练部分
"""
import pickle
from HGST import HGST
from numpy import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
#from torch.utils.tensorboard import SummaryWriter

#Path_to_log_dir = './datasets/tensorboard'
#writer = SummaryWriter(Path_to_log_dir)

with open('./datasets/Train_Dataset_for_HGST.file', 'rb') as fo:
    Dataset = pickle.load(fo)

use_cuda = torch.cuda.is_available()
print(torch.cuda.device_count())

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

setup_seed(10)


def Train(model, Dataset, Epoch, loss_func, optimizer):
    model.train()
    loss_set = []
    for epoch in range(Epoch):
        start_time = time.time()
        train_losses = []
        for step, (x, y) in enumerate(Dataset):
            #print(next(model.parameters()).device)
            #print(x.device)
            output = model(x)
            target = y[:, 0].unsqueeze(1)  # 预测inbound：0， 预测outbound：1
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.cpu().detach().numpy())
        # test_loss = Evaluate(model, Test_Dataset, loss_func)
        #writer.add_scalar('Train loss', mean(train_losses), epoch+1)
        #writer.flush()
        loss_set.append(mean(train_losses))
        end_time = time.time()
        cost = str(end_time - start_time)
        print('Epoch [{}/{}], Loss: {:.4f}, Time: {}'.format(epoch + 1, Epoch, mean(train_losses), cost))
    plt.plot(loss_set)
    # plt.show()
    plt.show()


model = HGST(lstm_in_dim=1,
             lstm_out_dim=32,
             dcnn_in_dim=10,
             dcnn_out_dim=2,
             dcnn_layers=2,
             FNN_in_dim=8,
             FNN_hidden1=128,
             FNN_hidden2=64)
model.double()

EPOCH = 100
LR = 0.000001
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
loss_func = nn.MSELoss()
model = model.cuda()
loss_func = loss_func.cuda()
Train(model, Dataset, EPOCH, loss_func, optimizer)

PATH = './trained_models/HGST_1_32_10_2_2_100_0000001.pth'
torch.save(model.state_dict(), PATH)

