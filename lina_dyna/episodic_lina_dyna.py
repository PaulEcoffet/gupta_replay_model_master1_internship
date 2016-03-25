# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:37:47 2016

@author: paul
"""

#%% Gestion des import

from __future__ import print_function
import numpy as np
from Queue import PriorityQueue
import networkx as nx
import matplotlib.pyplot as plt

def print_mat(mat):
    """
    Do a pretty print of a sparse matrix
    """
    print(" " * 3, end="")
    for j in range(len(mat[0])):
        print (j, end=" "*(5 - len(str(j))))

    print()
    for i, line in enumerate(mat):
        print(i, end=": ")
        for elem in line:
            if elem != 0:
                print("{:.2f}".format(elem), end=" ")
            else:
                print(" "*5, end="")
        print()

#%% Environment definition

INFTY = float("inf")

G_W = 5  # Grid width
G_H = 5  # Grid height
end = np.zeros((G_W * G_H, 1))
end[4 * G_W + 3, 0] = 1

action = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

def do_action(phi, a):
    pos = np.where(phi == 1)[0][0]
    coord = np.array([pos / G_W, pos % G_W])
    new_coord = coord + action[a]
    if not (0 <= new_coord[0] < G_H and 0 <= new_coord[1] < G_W):
        return phi, 0
    else:
        i = new_coord[0] * G_W + new_coord[1]
        phi_n = np.zeros((G_W * G_H, 1))
        phi_n[i, 0] = 1
        if np.array_equal(phi_n, end):
            return phi_n, 1
        else:
            return phi_n, 0
        
gamma = 0.9
alpha = 0.1

#%% episodic linear dyna Q-learning

def softmax(score, tau):
    exp_score = np.exp(score/tau)
    prob = exp_score/np.sum(exp_score)
    return np.random.choice(len(score), p=prob)


def ep_lindyn_mg(phi, theta, F, b):
    pqueue = PriorityQueue()
    q = np.array([0. for i in action])
    for day in range(10):
        for i in range(3):
            start_pos = np.random.randint(G_W * G_H)
            phi = np.zeros((G_W * G_H, 1))
            phi[start_pos, 0] = 1
            print ("starting at", start_pos)
            while not np.array_equal(phi, end):
                for a in range(len(action)):
                    q[a] = np.dot(b[a].T, phi) + gamma * np.dot(np.dot(theta.T, F[a]), phi)
                a = softmax(q, 0.1)
                phi_n, r = do_action(phi, a)
                #print (np.where(phi == 1), "->", np.where(phi_n == 1))
                delta = r + gamma * np.dot(theta.T, phi_n) - np.dot(theta.T, phi)
                theta = theta + alpha * delta * phi
                F[a] = F[a] + np.dot(alpha * (phi_n - np.dot(F[a], phi)), phi.T)
                b[a] = b[a] + alpha * (r - np.dot(b[a].T, phi)) * phi
                for i in range(len(phi)):
                    if phi[i] != 0:
                        pqueue.put((-np.abs(delta * phi[i]), i))
                phi = phi_n
            # end of episode
        print("sleeping")
        # Sleep
        
        G_sleep = nx.Graph()
        
        
        p = 100 # Number of replay max
        print("La queue est vide :", pqueue.empty())
        while not pqueue.empty() and p > 0:
            unused_prio, i = pqueue.get()
            #print("looking at", i)
            for j in range(F.shape[2]):
                if np.any(F[:, i, j] != 0):
                    #print (i, '->', j)
                    try:   
                        G_sleep[min(i, j)][max(i,j)]["weight"] += 1
                    except KeyError:
                        G_sleep.add_edge(min(i, j), max(i, j), weight=1)
                    delta = -INFTY
                    for a in range(len(action)):
                        delta = np.max((delta, b[a][j] + gamma * np.dot(theta.T, F[a, :, j]) - theta[j]))
                    theta[j] = theta[j] + alpha * delta
                    pqueue.put((-np.abs(delta), j))
            p = p - 1
        pos = nx.circular_layout(G_sleep)
        nx.draw_networkx_nodes(G_sleep, pos)
        max_w = 1
        for u,v, w in G_sleep.edges_iter(data="weight"):
            max_w = max(max_w, w)
        for u,v, w in G_sleep.edges_iter(data="weight"):
            nx.draw_networkx_edges(G_sleep, pos, edgelist=[(u, v)], width=float(w)/max_w * 10)
        nx.draw_networkx_labels(G_sleep,pos,font_size=10,font_family='sans-serif')
        plt.axis("off")
        plt.show()
    return theta, b, F


#%%%

start = np.zeros((G_W * G_H, 1))
start[0, 0] = 1

theta = np.zeros((G_W * G_H, 1))
F = np.zeros((len(action), G_W * G_H, G_W * G_H))
b = np.zeros((len(action), G_W * G_H, 1))

theta_f, b_f, F_f = (ep_lindyn_mg(start, theta, F, b))

t_shape = theta_f.reshape((G_H, G_W))

#%% F_plot du cheat

G = nx.Graph()
F_sum = np.sum(F, axis=0)
for i in xrange(F_sum.shape[0]):
    for j in xrange(i):
        weight=F_sum[i][j] + F_sum[j][i]
        if weight > 1e-14:
            G.add_edge(i, j, weight=weight)
#nx.draw_circular(G)

##################
###### TODO ######
##################
# Evaluer les performances en fin et début de journée
# Passer avec des champs récepteurs
# grapher F en mode bogoss
# Linear Dyna lambda ?