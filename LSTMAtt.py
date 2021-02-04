# _*_coding:utf-8_*_
# 作者    ： YiPeiyu
# 创建时间 : 2020/11/16 15:41
import torch.nn as nn
import torch
from data_loader import Get_All_Data
import torch.utils.data as Data
import math
import numpy as np
import torch.nn.functional as F

# SEED = 0
# torch.manual_seed(SEED)
#
# X_train_1, Y_train, X_test_1, Y_test, Y_test_original, a, b = Get_All_Data(TG=10,
#                                                                           time_lag=6,
#                                                                           TG_in_one_day=96,
#                                                                           forecast_day_number=7,
#                                                                           TG_in_one_week=672)
# X_train_1 = torch.from_numpy(X_train_1).cuda()
# Y_train = torch.from_numpy(Y_train).cuda()
# X_test_1 = torch.from_numpy(X_test_1).cuda()
# Y_test = torch.from_numpy(Y_test).cuda()
# Train_dataset = Data.TensorDataset(X_train_1, Y_train)
# Test_dataset = Data.TensorDataset(X_test_1, Y_test)


class LSTMAtt(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(LSTMAtt, self).__init__()
        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.lstm_r = nn.LSTM(in_dim, hidden_dim, num_layers=2)
        self.lstm_d = nn.LSTM(in_dim, hidden_dim, num_layers=2)
        self.lstm_w = nn.LSTM(in_dim, hidden_dim, num_layers=2)
        self.dropout = nn.Dropout(0.5)

        self.fc_r = nn.Linear(hidden_dim, 1)
        self.fc_d = nn.Linear(hidden_dim, 1)
        self.fc_w = nn.Linear(hidden_dim, 1)

        self.fc = nn.Linear(3, 1)

    # 注意力机制 软性 机制 k=value=x
    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn

    def forward(self, r_feature, d_feature, w_feature):
        out_r, (hidden_r, c_r) = self.lstm_r(r_feature)
        out_d, (hidden_d, c_d) = self.lstm_d(d_feature)
        out_w, (hidden_w, c_w) = self.lstm_w(w_feature)

        #recently
        out_r = out_r.permute(1, 0, 2)
        query_r = self.dropout(out_r)
        attn_out_r, attention_r = self.attention_net(out_r, query_r)
        logit_r = self.fc_r(attn_out_r)

        #daily
        out_d = out_d.permute(1, 0, 2)
        query_d = self.dropout(out_d)
        attn_out_d, attention_d = self.attention_net(out_d, query_d)
        logit_d = self.fc_d(attn_out_d)

        #weekly
        out_w = out_w.permute(1, 0, 2)
        query_w = self.dropout(out_w)
        attn_out_w, attention_w = self.attention_net(out_w, query_w)
        logit_w = self.fc_w(attn_out_w)

        #concatenate
        logit = torch.cat((logit_r, logit_d, logit_w), dim=1)
        logit = self.fc(logit)
        return logit
#
# LSTMAtt = LSTMAtt(1, 3)
# LSTMAtt.double()
# LSTMAtt.cuda()
# EPOCH = 50
# LR = 0.001
# optimizer = torch.optim.Adam(LSTMAtt.parameters(), lr=LR)
# loss_func = nn.L1Loss()
#
# train_loss = []
# for epoch in range(EPOCH):
#     losses = []
#     for step, (x, y) in enumerate(Train_dataset):
#         LSTMAtt.zero_grad()
#         LSTMAtt.train()
#         r_feature = x[:, 0:5].t().unsqueeze(-1)
#         d_feature = x[:, 5:10].t().unsqueeze(-1)
#         w_feature = x[:, 10:15].t().unsqueeze(-1)
#         out = LSTMAtt(r_feature, d_feature, w_feature)
#         loss = loss_func(out, y)
#         losses.append(loss)
#         loss.backward()
#         optimizer.step()
#     train_loss.append(np.sum(losses))
#     if (epoch + 1) % 5 == 0:
#         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, EPOCH, np.sum(losses)))








