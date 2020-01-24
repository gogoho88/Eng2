# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:38:35 2020

@author: owner
"""

import numpy as np

data_load = np.load('D://CJH/Research/En2/Data/Result_Atlas_v3.npz')

cost_training = data_load['d2_avg_cost_training']
cost_test = data_load['d2_avg_cost_test']/64
acc_training = data_load['d2_avg_acc_training']
acc_test = data_load['d2_avg_acc_test']


acc_test_sel = np.max(acc_test, axis=0)
best_midx = np.argmax(acc_test_sel)


d1_node_hidden1 = np.array([600,250,100])
d1_node_hidden2 = np.array([600,300,150])
d1_node_hidden3 = np.array([800,400,200])
d1_node_hidden4 = np.array([900,600,300])
droprate1 = np.array([0,0])
droprate2 = np.array([0.25,0.25])
droprate3 = np.array([0.5,0.5])
d1_noiselevel = np.array([0.2,0.1,0.05,0.01])

list_hidden = [d1_node_hidden1,d1_node_hidden2,d1_node_hidden3,d1_node_hidden4]
list_drop = [droprate1,droprate2,droprate3]
list_noiselevel = d1_noiselevel


m_idx = best_midx

i = int(m_idx/(len(list_drop)*len(list_noiselevel)))
j = int((m_idx%(len(list_drop)*len(list_noiselevel)))/len(list_noiselevel))
k = m_idx%len(list_noiselevel)

