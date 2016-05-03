# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:09:02 2016

@author: paul
"""

#%%

import numpy as np
import matplotlib.pyplot as pyplot

p = 0.4
eps = 1e-8

#%%

def action(state):
    return np.array(xrange(1, min(state, 100-state)+1), dtype=int)

def reward(state):
    return np.where(state == 100, 1, 0)

def score(s, p, a, V):
    return np.array(p * (reward(s + a) + V[s + a]) + (1 - p) * (reward(s - a) + V[s - a]))

def compute_V():
    V = np.array([0 for i in xrange(101)], dtype=float)

    delta = 1
    while delta > eps:
        delta = 0
        for s in xrange(1, 100):
            v = V[s]
            a = action(s)
            V[s] = np.max(score(s, p, a, V))
            delta = max(delta, abs(v - V[s]))
        pyplot.plot(V)

    return V

def compute_pi(V):
    pi = np.array([0 for i in V])
    for s in xrange(1, 100):
        a = action(s)
        pi[s] = a[np.argmax(score(s, p, a, V))]
    return pi


#%%
V = compute_V()
pi = compute_pi(V)