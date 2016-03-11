
"""
Created on Sun Feb 28 14:26:02 2016

@author: paul
"""
#%%
import numpy as np
import matplotlib.pyplot as pl

moves = np.array([(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1),
                  (-1, 1), (1, -1), (0, 0)])
m_txt = [">", "v", "^", "<", "b", "p", "d", "q", "*"]
wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
alpha = 0.1
inf = float("inf")


def reward(state):
    return -1


def new_state(s, a):
    s_n = s + moves[a]
    if not 0 <= s_n[0] < 7 or not 0 <= s_n[1] < 10:
        raise ValueError(s, a)
    if wind[s_n[1]] != 0:
        cwind = np.random.randint(wind[s_n[1]] - 1, wind[s_n[1]] + 1)
        s_n[0] = max(0, s_n[0] - cwind)
    return s_n

#%% Q-Learning

Q = np.array([[[0 if 0 <= m_col + col < 10 and 0 <= m_row + row < 7 else -inf
                for m_row, m_col in moves]
               for col in range(10)]
              for row in range(7)])

for i in range(1000):
    s = np.array((3, 0))
    a = np.argmax(Q[tuple(s)])
    while not (s[0] == 3 and s[1] == 7):
        s_n = new_state(s, a)
        r = reward(s_n)
        a_n = np.argmax(Q[tuple(s_n)])
        Q[tuple(s)][a] = Q[tuple(s)][a] + alpha * (r + Q[tuple(s_n)][a_n] - Q[tuple(s)][a])
        s = s_n
        a = a_n


#%% Using Q

grid = [[" " for j in range(10)] for i in range(7)]

s = np.array((3, 0))
grid[s[0]][s[1]] = "*"
a = np.argmax(Q[tuple(s)])
while not (s[0] == 3 and s[1] == 7):
    print s, moves[a], "->", s + moves[a]
    s_n = new_state(s, a)
    r = reward(s_n)
    a_n = np.argmax(Q[tuple(s_n)])
    Q[tuple(s)][a] = Q[tuple(s)][a] + alpha * (r + Q[tuple(s_n)][a_n] - Q[tuple(s)][a])
    grid[s_n[0]][s_n[1]] = m_txt[a_n]
    s = s_n
    a = a_n


for line in grid:
    print "|".join(line)
    print "-"+"+-"*(len(line)-1) + "+"
