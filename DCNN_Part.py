#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/11/18 10:30
@desc:DCNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# with open('./data/Train_DCNN_dataset.file', 'rb') as fo:
#     Train_dataset = pickle.load(fo)
# with open('./data/Test_DCNN_dataset.file', 'rb') as fo:
#     Test_dataset = pickle.load(fo)
# with open('./data/Train_DCNN_support.file', 'rb') as fo:
#     train_support = pickle.load(fo)
# with open('./data/Test_DCNN_support.file', 'rb') as fo:
#     test_support = pickle.load(fo)


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, A, x):
        #x = x.float()  # doule转float
        x = torch.einsum('tk,kq->tq', A, x)
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:  # a 1*76*76
            a = torch.squeeze(a, 0)  # a 76*76
            x1 = self.nconv(a, x)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(a, x1)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = h.unsqueeze(2)
        h = h.unsqueeze(3)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = h.squeeze(3)
        h = h.squeeze(2)
        return h


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, layers):
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        self.gconvs = nn.ModuleList()

        for i in range(layers):
            if i >= 1:
                self.gconvs.append(gcn(c_in=out_dim, c_out=out_dim, dropout=0.3))
            else:
                self.gconvs.append(gcn(c_in=in_dim, c_out=out_dim, dropout=0.3))

    def forward(self, x, supports):
        for i in range(self.layers):
            h = self.gconvs[i](x, supports)
            x = h
        return h


# in_dim = 10
# out_dim = 2
# layer = 1
# DCNN_model = GCNLayer(10, 2, 1)
# DCNN_model.double()
# print(DCNN_model)
#
# EPOCH = 20
# LR = 0.001
# optimizer = torch.optim.Adam(DCNN_model.parameters(), lr=LR)
# loss_func = nn.L1Loss()
#
# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(Train_dataset):
#         DCNN_model.zero_grad()
#         DCNN_model.train()
#         out = DCNN_model(x, train_support[step])
#         print(out)