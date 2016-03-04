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


G_W = 10  # Grid width
G_H = 10  # Grid height
starts = np.array([(3, 4), (4, 5), (0, 0), (7, 2)])
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


#%% defining policy

class Policy:

    def __init__(self):
        self.path = {(3, 4): [RIGHT, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, DOWN,
                              DOWN, DOWN, DOWN, DOWN, LEFT, DOWN, LEFT,
                              LEFT, UP, RIGHT, UP, LEFT],
                     (4, 5): [RIGHT, RIGHT, DOWN, LEFT]}

        self.action_index = 0
        self.episode_start = None

    def take_action(self, state):
        if not self.episode_start:
            self.episode_start = tuple(state)
            self.action_index = 0

        cur_action = self.path[self.episode_start][self.action_index]

        self.action_index += 1
        if self.action_index >= len(self.path[self.episode_start]):
            self.episode_start = None

        return cur_action



#%% Trigger TD Learning

policy = Policy()

pi = policy.take_action
TD_V = td_learn(pi, 10)



#%% Matrix comparison

np.linalg.norm(TD_V - True_V)