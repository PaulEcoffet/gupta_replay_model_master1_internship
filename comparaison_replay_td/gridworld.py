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

seed = 32

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
    
    def vec_take_action(self, state):
        grid_coord = vec_to_grid(state)
        return self.take_action(grid_coord)

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


#%% Grid to vec

def grid_to_vec(*in_coord):
    try:
        coord = (in_coord[0][0], in_coord[0][1])
    except TypeError:
        coord = (in_coord[0], in_coord[1])
    vec = np.zeros((G_H * G_W, 1))
    vec[coord[0] * G_W + coord[1]] = 1
    return vec

def vec_to_grid(vec):
    coord = np.where(vec == 1)
    coord = int(coord[0])
    return np.array((coord / G_W, coord % G_W))

#%% env definition

end_vec = grid_to_vec(end)
start_vec = grid_to_vec(starts[1])

class Env():
    def __init__(self):
        self.s = np.copy(start_vec)
        self.punish = False
    
    @property
    def reward(self):
        if self.is_end:
            return 1
        elif self.punish:
            return -1
        else:
            return 0
    
    def do_action(self, a):
        self.punish = False
        grid = vec_to_grid(self.s)
        new_coord = grid + action[a]
        if 0 <= new_coord[0] < G_H and 0 <= new_coord[1] < G_W:
            s_n = grid_to_vec(new_coord)
            self.s = s_n
        else:
            self.punish = True
    
    @property
    def gamma(self):
        if self.is_end:
            return 0
        return 0.9
    
    @property
    def is_end(self):
        return np.array_equal(self.s, end_vec)
        
        

#%% Forgetful LSTD(lambda)

def forgetful_lstd(env, pi, alpha, beta, ld, k, theta_init, d_init, A_init):
    feat_dim = (env.s.shape[0], 1)
    e = np.zeros(feat_dim)
    theta = np.copy(theta_init)
    d = np.copy(d_init)
    A = np.copy(A_init)
    phi = env.s
    for i in xrange(100):
        env = Env()
        phi = env.s
        while not env.is_end:
            assert(e.shape == feat_dim)
            assert(theta.shape == feat_dim)
            assert(d.shape == feat_dim)
            assert(A.shape == (G_W * G_H, G_W * G_H))
            a = pi(phi)
            env.do_action(a)
            phi_next = env.s
            r = env.reward
            gamma = env.gamma
            i_beta_phi = np.eye(feat_dim[0]) - beta * (np.dot(phi, phi.T))
            e = np.dot(i_beta_phi, e) + phi
            phi_phi_n = phi - gamma * phi_next
            A = np.dot(i_beta_phi, A) + np.dot(e, phi_phi_n.T)
            d = np.dot(i_beta_phi, d) + e * r
            e = gamma * ld * e
            for i in xrange(k):
                theta = theta + alpha * (d - np.dot(A, theta))
            phi = phi_next
    return theta

#%% run forgetful lstd(ld)

env = Env()
policy = Policy(seed)
pi = policy.vec_take_action
theta = np.zeros((G_H * G_W, 1))
d_init = theta / alpha # zeros
A_init = np.eye(G_H * G_W) / alpha

theta = forgetful_lstd(env, pi, alpha, alpha, 0, 10, theta, d_init, A_init)
theta_joli = theta.reshape((G_H, G_W))

print_mat(theta_joli)

#%% Forgetful sarsa

def choose_action(theta, phi, eps):
    if np.random.uniform() < eps:
        return np.random.choice(xrange(theta.shape[0]))
    else:
        return np.argmax([np.dot(theta[i].T, phi) for i in xrange(theta.shape[0])])

def forgetful_sarsa(env, alpha, beta, ld, k, theta_init, d_init, A_init, 
                    eps=0.01):
    env = Env()
    feat_dim = (env.s.shape[0], 1)
    e = np.zeros(feat_dim)
    theta = np.copy(theta_init)
    d = np.copy(d_init)
    A = np.copy(A_init)
    phi = env.s
    a = choose_action(theta, phi, eps)
    path = np.array([[" " for i in range(G_W)] for j in range(G_H)])
    while not env.is_end:
        env.do_action(a)
        r = env.reward
        phi_n = env.s
        path[vec_to_grid(phi)] = "*"
        a_n = choose_action(theta, phi_n, eps)
        print(vec_to_grid(phi), a, r)
        gamma = env.gamma
        i_beta_phi = np.eye(feat_dim[0]) - beta * (np.dot(phi, phi.T))
        e = np.dot(i_beta_phi, e) + phi
        phi_phi_n = phi - gamma * phi_n
        A[a] = np.dot(i_beta_phi, A[a]) + np.dot(e, phi_phi_n.T)
        d[a] = np.dot(i_beta_phi, d[a]) + e * r
        e = gamma * ld * e
        for i in xrange(k):
            theta[a] = theta[a] + alpha*(d[a] - np.dot(A[a], theta[a]))
        a = a_n
        phi = phi_n
    print(path)
    return theta

#%% Run forgetful_sarsa

env = Env()
theta = np.zeros((4, G_H * G_W, 1))
d_init = theta / alpha # zeros
A_init = np.array([np.eye(G_H * G_W) / alpha for i in range(4)])

theta = forgetful_sarsa(env, alpha, alpha, 0, 20, theta, d_init, A_init)