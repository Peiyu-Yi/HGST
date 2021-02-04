#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/19 16:21
@desc: HGST: hybrid graph-based spatial-temporal model
"""
from LSTMAtt import LSTMAtt
from DCNN_Part import GCNLayer
from FNN_Part import FNN
import torch
import torch.utils.data as Data
import torch.nn as nn
from numpy import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
# SEED = 0
# torch.manual_seed(SEED)
with open('./datasets/Dataset_for_HGST.file', 'rb') as fo:
    Dataset = pickle.load(fo)
with open('./datasets/Test_Dataset_for_HGST.file', 'rb') as fo:
    Test_Dataset = pickle.load(fo)
Dataset_loader = Data.DataLoader(Dataset, batch_size=3, shuffle=False)
#
# for batch_ndx, sample in enumerate(Dataset_loader):
#     a = batch_ndx
#     b = sample


class HGST(nn.Module):
    def __init__(self, lstm_in_dim, lstm_out_dim, dcnn_in_dim, dcnn_out_dim, dcnn_layers, FNN_in_dim, FNN_hidden1, FNN_hidden2):
        super(HGST, self).__init__()
        self.LSTM = LSTMAtt(lstm_in_dim, lstm_out_dim)
        self.FNN = FNN(FNN_in_dim, FNN_hidden1, FNN_hidden2)
        self.DCNN = GCNLayer(dcnn_in_dim, dcnn_out_dim, dcnn_layers)
        self.linear = nn.Linear(3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        r_feature = x[:, 0:5].transpose(0, 1).unsqueeze(-1)
        d_feature = x[:, 5:10].transpose(0, 1).unsqueeze(-1)
        w_feature = x[:, 10:15].transpose(0, 1).unsqueeze(-1)
        dcnn_x = x[:, 15:25]
        FNN_x = x[:, 25:33]
        support_1 = x[:, 33: 109].unsqueeze(0)
        support_2 = x[:, 109: 185].unsqueeze(0)
        support = torch.stack(tuple([support_1, support_2]), dim=0)
        #r_feature_cuda = r_feature.cuda()
        #d_feature_cuda = d_feature.cuda()
        #w_feature_cuda = w_feature.cuda()
        #dcnn_x_cuda = dcnn_x.cuda()
        #support_cuda = support.cuda()
        O_temp_in = self.LSTM(r_feature, d_feature, w_feature)
        O_external_in = self.FNN(FNN_x)
        O_spat = self.DCNN(dcnn_x, support)
        O_spat_in = O_spat[:, 0].unsqueeze(1)  # in：0，  out：1
        Out = torch.cat((O_temp_in, O_spat_in, O_external_in), dim=1)
        Out = self.linear(Out)
        Out = self.relu(Out)
        return Out




# model = HGST(lstm_in_dim=1,
#              lstm_out_dim=3,
#              dcnn_in_dim=10,
#              dcnn_out_dim=2,
#              dcnn_layers=1)
# model.double()
# model_cuda = model.cuda()
# EPOCH = 100
# LR = 0.0001
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# loss_func = nn.MSELoss()
#
#
#
#
# train_losses = []
# test_losses = []
# for epoch in range(EPOCH):
#     losses = []
#     test_loss = []
#     for step, (x, y) in enumerate(Dataset):
#         output = model_cuda(x)
#         target_cuda = y[:, 0].unsqueeze(1).cuda()
#         #target2 = torch.from_numpy(target).unsqueeze(1)
#         loss = loss_func(output, target_cuda)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.cpu().detach().numpy())
#     test_loss = Evaluate(Test_Dataset, loss_func)
#     if (epoch + 1) % 1 == 0:
#         print('Epoch [{}/{}], Loss: {:.4f}, Test loss:{:.4f}'.format(epoch + 1, EPOCH, mean(losses), test_loss))
#     train_losses.append(mean(losses))
#     test_losses.append(test_loss)
#
#
# plt.subplot(211)
# plt.plot(train_losses)
# plt.subplot(212)
# plt.plot(test_losses)
# plt.show()