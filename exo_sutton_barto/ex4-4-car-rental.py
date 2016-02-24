# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 18:24:37 2016

@author: paul
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial

gamma = 0.9

lt1 = 3
lt2 = 4
lr1 = 3
lr2 = 2

States = [(i, j) for i in range (0, 21) for j in range(0, 21)]

#%% calcul proba poisson

def dp(l, n):
    return np.power(l,n)/factorial(n) * np.exp(-l)


pt1 = dp(lt1, np.array(range(21)))
pt1[-1] = 1 - np.sum(pt1[:-1])
pr1 = dp(lr1, np.array(range(21)))
pr1[-1] = 1 - np.sum(pr1[:-1])

pt2 = dp(lt2, np.array(range(21)))
pt2[-1] = 1 - np.sum(pt2[:-1])
pr2 = dp(lr2, np.array(range(21)))
pr2[-1] = 1 - np.sum(pr2[:-1])


#%% Calcul proba complète et reward

def proba_reward(state, future_state, action):
    future = (future_state[0] - action, future_state[1] + action)
    if future[0] > 20 or future[1] > 20 or future[0] < 0 or future[1] < 0 or abs(action) > 5:
        return 0, 0
    else:
        proba = 0
        reward = 0
        cur_pt1 = pt1[:state[0] + 1]
        cur_pt1[state[0]+1] = 1 - np.sum(cur_pt1[:state[0]+1])
        
        cars_back = future[0] - state[0]
        cur_pr1 = np.zeros(state[0])
        
        
        for i in range(state[0] + 1):
            for j in range(state[1] + 1):
                cur_proba = dp(lt1, i) * dp(lr1, future[0] - state[0] + i)
                cur_proba *= dp(lt2, j) * dp(lr2, future[1] - state[1] + j)
                reward += ((i + j) * 10 + 2*abs(action)) * cur_proba
                proba += cur_proba
        return proba, reward

def reward(state, future_state, action):  
    return (tooked1 + tooked2) * 10 - 2*abs(action)

def evaluate_policy(pi):
    V = [[0 for i in range(0, 21)] for j in range(0, 21)]
    delta = 1
    while delta > 0.01:
        delta = 0
        for (i, j) in States:
            v = V[i][j]
            tmp = 0
            for future in States:
                proba, reward = proba_reward((i, j), future, pi[i][j])
                tmp += proba * (reward + gamma * V[future[0]][future[1]])
            V[i][j] = tmp
            delta = max(delta, abs(v - V[i][j]))
            print(i*20+j)
        print(V)
        print ""
        raw_input()
    return V

pi = [[1 for j in range(21)] for i in range(20)]

print(evaluate_policy(pi))

            


