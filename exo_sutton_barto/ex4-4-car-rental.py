# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 18:24:37 2016

@author: paul
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial
import sys

gamma = 0.9

lt1 = 3
lt2 = 4
lr1 = 3
lr2 = 2

lt = [lt1, lt2]
lr = [lr1, lr2]

States = [(i, j) for i in range (0, 21) for j in range(0, 21)]

#%% Joli print
def printMatrix(testMatrix):
    print ' ',
    for i in range(len(testMatrix[1])):  # Make it work with non square matrices.
          print i,
    print
    for i, element in enumerate(testMatrix):
          print i, ' '.join([str(x) for x in element])
    sys.stdout.flush()


#%% calcul proba poisson

def dp(l, n):
    return np.where(n >= 0, np.power(l,n)/factorial(n) * np.exp(-l), 0)


pt1 = dp(lt1, np.array(range(21)))
pt1[-1] = 1 - np.sum(pt1[:-1])
pr1 = dp(lr1, np.array(range(42)))
pr1[-1] = 1 - np.sum(pr1[:-1])

pt2 = dp(lt2, np.array(range(21)))
pt2[-1] = 1 - np.sum(pt2[:-1])
pr2 = dp(lr2, np.array(range(42)))
pr2[-1] = 1 - np.sum(pr2[:-1])


#%% Calcul proba complète et reward

def proba_reward_vecto(state, future_state, action):
    future = (future_state[0] + action, future_state[1] - action)
    if future[0] > 20 or future[1] > 20 or future[0] < 0 or future[1] < 0 or abs(action) > 5:
        return 0, 0
    else:
        proba = 0
        reward = 0
        cur_pt1 = np.zeros(21)
        cur_pt1[:state[0]+1] = pt1[:state[0] + 1]
        cur_pt1[state[0]] = 1 - np.sum(cur_pt1[:state[0]])
        
        cars_back = future[0] - state[0]
        cur_pr1 = np.zeros(21)
        start = max(0, -cars_back)
        cur_pr1[start:] = pr1[max(0, cars_back):max(0, cars_back)+21-start]
        print state, future
        print cur_pt1
        print cur_pr1
        
        cur_pt2 = np.zeros(21)
        cur_pt2[:state[1] + 1] = pt2[:state[1] + 1]
        cur_pt2[state[1]] = 1 - np.sum(cur_pt2[:state[1]])
        
        cars_back = future[1] - state[1]
        cur_pr2 = np.zeros(21)
        start = max(0, -cars_back)
        cur_pr2[start:] = pr2[max(0, cars_back):max(0, cars_back)+21-start]
  
        proba1 = cur_pt1 * cur_pr1
        proba2 = cur_pt2 * cur_pr2
        reward = (np.array(range(21)).dot(proba1) + np.array(range(21)).dot(proba2)) * 10 - 2 * abs(action)
        proba_trans = np.sum(proba1) * np.sum(proba2)
        return proba_trans, reward

def proba_reward_iter(state, future_state, action):
    future = (future_state[0] + action, future_state[1] - action)
    if future[0] > 20 or future[1] > 20 or future[0] < 0 or future[1] < 0 or abs(action) > 5:
        return 0, 0
    else:
        proba = [0, 0]
        reward = [0, 0]
        
        for stand in (0, 1):
            for cars_tooked in range(state[stand]+1):
                cars_returned = future[stand] - state[stand] + cars_tooked
                cur_proba = dp(lt[stand], cars_tooked) * dp(lr[stand], cars_returned)
                reward[stand] += 10 * cars_tooked * cur_proba
                proba[stand] += cur_proba
                #print "stand:", stand, ", tooked:", cars_tooked, ", returned:", cars_returned
                #print "proba:", cur_proba, "reward:", 10 * cars_tooked * cur_proba
        
        return proba[0] * proba[1], np.sum(reward) - 2 * abs(action)


proba_reward = proba_reward_iter

#%% Calcul des policies

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
    return V

pi = [[0 for j in range(21)] for i in range(21)]


def improve_policy(pi):
    policy_stable = False
    itera = 1
    while not policy_stable:
        print "*"*40
        print "iter:", itera
        print "*"*40
        printMatrix(pi)
        policy_stable = True
        V = evaluate_policy(pi)
        for (i, j) in States:
            b = pi[i][j]
            best = None
            best_score = -1e128
            for action in range(-5, 6):
                tmp = 0
                for future in States:
                    proba, reward = proba_reward((i, j), future, action)
                    tmp += proba * (reward + gamma * V[future[0]][future[1]])
                if tmp > best_score:
                    best_score = tmp
                    best = action
            pi[i][j] = best
            if b != pi[i][j]:
                policy_stable = False
        itera += 1
    return pi


#%% Run da shit
         
#printMatrix(evaluate_policy(pi))
printMatrix(improve_policy(pi))

