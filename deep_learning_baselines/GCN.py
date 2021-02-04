#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/12/5 13:22
@desc: 对比模型：GCN 2层
"""
import torch
import torch.nn as nn
import pickle
import dgl
from dgl.nn.pytorch import GraphConv
import networkx as nx
import matplotlib.pyplot as plt


with open('./adjacency_gcn.file', 'rb') as fo:
    adj_matrix = pickle.load(fo)


URTNetwork = dgl.DGLGraph(adj_matrix)
#nx.draw(URTNetwork.to_networkx(), with_labels=True)
#plt.show()

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim1, hidden_dim2)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x[:, 15: 25]
        #x = x.clone().to(torch.float)
        h = self.conv1(URTNetwork, x)
        #h = self.conv2(URTNetwork, h)
        Out = self.linear(h)
        return Out