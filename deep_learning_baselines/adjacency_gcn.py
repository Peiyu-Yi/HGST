#  生成URTNetwork的邻接矩阵 76个站点
import numpy as np
import pandas as pd
import pickle
import pandas as pd

adj = np.zeros([110, 110], dtype=int)

# line 1
adj[0][1] = 1
adj[1][2] = 1
adj[2][3] = 1
adj[3][4] = 1
adj[4][5] = 1
adj[5][6] = 1
adj[6][7] = 1
adj[7][8] = 1
#adj[8][10] = 1
adj[10][11] = 1
adj[11][12] = 1
adj[12][13] = 1
adj[13][14] = 1
adj[18][19] = 1
adj[19][20] = 1
adj[20][21] = 1
# line 2
adj[22][23] = 1
adj[30][31] = 1
adj[31][32] = 1
adj[32][33] = 1
adj[33][34] = 1
adj[36][37] = 1
adj[37][38] = 1
adj[38][39] = 1

# line 3
adj[53][54] = 1
adj[56][57] = 1
adj[57][58] = 1
adj[59][60] = 1
adj[60][61] = 1
adj[63][64] = 1
adj[64][65] = 1
adj[65][66] = 1
adj[66][67] = 1
adj[67][68] = 1
adj[68][69] = 1
adj[69][70] = 1
adj[71][72] = 1
adj[72][73] = 1
adj[73][74] = 1
adj[74][75] = 1
adj[75][76] = 1
adj[76][77] = 1
adj[77][78] = 1
adj[81][82] = 1
adj[82][83] = 1
adj[83][84] = 1

# line 4
adj[89][90] = 1
adj[92][93] = 1
adj[93][94] = 1
adj[94][95] = 1
adj[95][96] = 1
adj[96][97] = 1
adj[97][98] = 1
adj[98][99] = 1
adj[99][100] = 1

#  cross
adj[1][90] = 1
adj[1][23] = 1
adj[0][22] = 1
adj[0][89] = 1
adj[2][22] = 1
adj[2][63] = 1
adj[3][63] = 1
adj[3][30] = 1
adj[5][30] = 1
adj[4][31] = 1
adj[67][94] = 1
adj[67][95] = 1

top_index = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,16,18,19,20,21
             ,22,23,30,31,32,33,34,36,37,38,39,46,48,51,53,54,56
              ,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,73,74,75
              ,76,77,78,81,82,83,84,87,89,90,92,93,94,95,96,97,98,99
              ,100,104,107,109]
index_to_delete = [9, 15, 17, 24, 25, 26, 27, 28, 29, 35, 40, 41, 42, 43, 44, 45, 47, 49, 50, 52, 55, 62, 79, 80, 85,
                   86, 88, 91, 101, 102, 103, 105, 106, 108]
adj = np.delete(adj, index_to_delete, axis=0)
adj = np.delete(adj, index_to_delete, axis=1)

#with open('./adjacency_gcn.csv', 'wb') as fo:
#  pickle.dump(adj, fo)
np.savetxt('./dajacency.csv', adj, fmt='%f', delimiter=',')