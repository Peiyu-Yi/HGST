#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2020/12/7 12:50
@desc: 生成ConvGCN的数据
"""
import csv
import pandas as pd
import numpy as np

top_index = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,16,18,19,20,21
             ,22,23,30,31,32,33,34,36,37,38,39,46,48,51,53,54,56
              ,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,73,74,75
              ,76,77,78,81,82,83,84,87,89,90,92,93,94,95,96,97,98,99
              ,100,104,107,109]

inedx_delete = [9, 15, 17, 24, 25, 26, 27, 28, 29, 35, 40, 41, 42, 43, 44, 45, 47, 49, 50, 52,
                55, 62, 79, 80, 85, 86, 88, 91, 101, 102, 103, 105, 106, 108, 110]

path = '../data/201501_inflow_30min_data.csv'
f = pd.read_csv(path)
f.drop(f.index[inedx_delete], inplace=True)
f.to_csv('in_30min.csv', index=False)