# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:00:40 2016

@author: paul
"""

#%% Gestion des import

from __future__ import print_function
import numpy as np

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

seed = 1000

G_W = 10  # Grid width
G_H = 10  # Grid height
starts = np.array([(0, 0), (4, 5), (3, 4), (7, 2)])
end = np.array((5, 6))

action = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

# learning constants

gamma = 0.9
alpha = 0.9


def r(state):
    """
    The reward function
    """
    if np.array_equal(state, end):
        return 1
    else:
        return 0


True_V = [[gamma**(abs(i - end[0]) + abs(j - end[1])) for j in range(G_W)]
          for i in range(G_H)]


#%% TD_learning definition

def td_learn(pi, nb_episodes, ld=0):
    """
    Compute the values V of the states according to the policy `pi` with
    a lambda `ld` doing `nb_episodes` episodes
    pi - the policy to be evaluated, a function
    ld - the lambda of the TD(lambda)
    """
    V = np.array([[0 for j in range(G_W)] for i in range(G_H)], dtype=float)
    for c_episode in range(nb_episodes):
        s = starts[0]
        while not np.array_equal(s, end):
            a = pi(s)
            s_next = s + action[a]
            V[tuple(s)] = V[tuple(s)] + alpha * (r(s_next) + gamma * V[tuple(s_next)]
                                      - V[tuple(s)])
            s = s_next
    return V


#%% Learning by replay ; Algo 3 -> 6 (Van Seijen & Sutton; 2015)
# http://webdocs.cs.ualberta.ca/~vanseije/resources/papers/vanseijenicml15.pdf
# It is super slow, takes a lot of memory and do a lot of redundant computation


def do_action(s, a):
    s_n = s + action[a]
    gamma = 0.9
    if np.array_equal(s_n, end):
        gamma = 0
    return s_n, gamma, r(s_n)

def plan_by_replay(V_init, alpha, k, m, pi):
    """
    V_init - Initial guess of the state values
    alpha - learning step
    k - number of replays after taking action
    m - number of previous states to take into account before replay
    pi - the policy, a function
    """
    V = np.copy(V_init)
    Phi = []
    Y = []
    W = []
    s = starts[0]
    while not np.array_equal(s, end):
        a = pi(s)
        s_n, gamma, r = do_action(s, a)
        Phi.append(s)
        Y.append((r, gamma, s_n))
        W.append(np.copy(V))
        for i_replay in range(k):
            W = update_weights(W, V, m)
            U = compute_targets(Y, W)
            V = replay(Phi, U, alpha, V_init)
        s = s_n
    return V

def update_weights(W, V, m):
    t = len(W)
    return [np.copy(W[i]) if i < t-m else np.copy(V) for i in range(t)]

def compute_targets(Y, W):
    U = np.zeros((len(Y),))
    for i in range(len(Y)):
        r, gamma, s_n = Y[i]
        U[i] = r + gamma * W[i][tuple(s_n)]
    return U

def replay(Phi, U, alpha, V_init):
    V = np.copy(V_init)
    for i in range(len(Phi)):
        V[tuple(Phi[i])] += alpha * (U[i] - V[tuple(Phi[i])])
    return V
        

#%% defining policy

class Policy:

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def take_action(self, state):
        future_state = np.array((-1, -1))
        cur_action = -3
        while not (0 <= future_state[0] < G_H and 0 <= future_state[1] < G_W):
            cur_action = self.rng.randint(4)
            future_state = state + action[cur_action]
        return cur_action

#%% Trigger TD Learning with 10 episodes

policy = Policy(seed)

pi = policy.take_action
TD_V = td_learn(pi, 10)

#%% Trigger Planning by replay learning with 10 episodes, no memory and replay
# aka TD(0) but way slower
policy = Policy(seed)

pi = policy.take_action
V = np.zeros((G_H, G_W))
RP_V = V
for i in range(10):
    RP_V = plan_by_replay(RP_V, alpha, 1, 1, pi)


#%% Matrix comparison, they are effectively equivalent

print("With 10 episodes and same policy")
print("euclidian distance between True and TD V:", np.linalg.norm(TD_V - True_V))
print("euclidian distance between True and RP V:", np.linalg.norm(RP_V - True_V))
print("TD_V and RP_V are equal:", np.array_equal(RP_V, TD_V))

#%% Doing some replay

policy = Policy(seed)

pi = policy.take_action
V = np.zeros((G_H, G_W))
RP3_V = V
for i in range(4):
    RP3_V = plan_by_replay(RP3_V, alpha, 3, 3, pi)

print("euclidian distance between True and TD V:", np.linalg.norm(TD_V - True_V))
print("euclidian distance between True and RP3 V:", np.linalg.norm(RP3_V - True_V))

print ("RP3 is better than TD(0):", np.linalg.norm(TD_V - True_V) > np.linalg.norm(RP3_V - True_V))
