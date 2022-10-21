#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import copy
from pandas import *
import scipy.optimize as sopt
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.optimize import minimize_scalar, minimize, root
from scipy.optimize import bisect
np.set_printoptions(precision=3)


# ### Import Data

# In[64]:


link_data = pd.read_excel('link_data.xlsx')
OD_matrix = pd.read_excel('O-D_matrix.xlsx')
link_data['cap'] = link_data['cap'] * 1000
#print(link_data)
print(OD_matrix)


# ### Functions Defined

# In[65]:


def tt_func(df):
    t0 = np.array(df.t0)
    cap = np.array(df.cap)
    x = np.array(df.x)
    return np.multiply(t0,(1+ 0.15*(np.power(np.divide(x,cap),4))))

def convergence(x_n,x_n1):
    return np.sqrt(np.sum(np.power(np.subtract(x_n1,x_n),2))) / np.sum(x_n)

def return_tt(df):
    x_n = np.array(df['x'])
    t0 = np.array(df.t0)
    ca = np.array(df.cap)
    obj =  np.sum(np.multiply(t0,np.add(x_n,0.15*np.divide(np.power(x_n,4),np.power(ca,4)))))
    return obj

def obj_func(alpha,df):
    y = np.array(df.y)
    x_n = np.array(df.x)
    t0 = np.array(df.t0)
    ca = np.array(df.cap)
    x_sub = np.add(x_n,alpha * np.subtract(y,x_n))
    obj = np.sum(np.multiply(t0,np.add(x_sub,(0.15/5)*np.divide(np.power(x_sub,5),np.power(ca,4)))))
    return obj


# ### Function for Dijkstra

# In[67]:


def dijkstra_sp(orig,dest,df):
    G1=nx.DiGraph()
    #list_nodes=[1,2,3,4,5,6,7,8,9,10,
              # 11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    #G1.add_nodes_from(list_nodes)
    G1.add_edges_from([(link_data['O'][i],link_data['D'][i],{'dist': link_data['tt'][i]}) for i in link_data.index])
    path = nx.dijkstra_path(G1, source=orig, target=dest)
    return path


# ### UE 

# In[68]:


### start with intitialization
## Initialization
link_data['x'] = 0
link_data['y'] = 0
Z = []
T=[]
conv_test = 100
loop = 1
cap = link_data['cap']
while conv_test > 1e-7:
    print('Step : ', loop)
    link_data['y'] = 0
    x_n = np.array(link_data['x'])
    link_data['tt'] = tt_func(link_data)
    for i in range (len(OD_matrix)):
        orig=OD_matrix['O'][i]
        dest=OD_matrix['D'][i]
        fl=OD_matrix['Q'][i]
        path=dijkstra_sp(orig,dest,link_data)
        for k in range(len(path)-1):
            d_new = path[k] 
            o_new = path[k+1]
            link_data.loc[(link_data.O==o_new) & (link_data.D==d_new),'y'] +=fl
    if loop!= 1:
        alpha = minimize_scalar(obj_func,args=(link_data), bounds=(0.0, 1.0),method='bounded').x
    else :
        alpha = 1
        #pred1 = pred
    print("Alpha : ",alpha)
    print("Z : ",obj_func(alpha,link_data))
    Z.append(obj_func(alpha,link_data))
    y = np.array(link_data['y'])
    #print(x_n,y)
    x_n1 = x_n + alpha * (y-x_n)
    link_data['x'] = x_n1 ## Update x to update t
    #print("T : ",return_tt(link_data))
    #T.append(return_tt(link_data))
    if loop != 1:
        conv_test= convergence(x_n,x_n1)
        print("Conv : ",conv_test)
    loop += 1
link_data['ratio'] = link_data['x'] / link_data['cap']
        


# In[62]:


# for i in range (len(OD_matrix)):
#         orig=OD_matrix['O'][i]
#         dest=OD_matrix['D'][i]
#         fl=OD_matrix['Q'][i]
#         print(orig,dest,fl)


# In[ ]:




