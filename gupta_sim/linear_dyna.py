# -*- coding: utf8 -*-


import numpy as np
from Queue import PriorityQueue


gamma = 0.9
alpha = 0.1

def softmax(score, tau, straight_bias=False):
    """
    get what action to do with a combination of a softmax and a straight bias
    adapted from Minija Tamosiunaite et al. (2008) if `straight_bias` is True.

    Note that I no longer use a straight bias which lead to weird behaviours and
    I train the agent in a TrainingEnvironment instead which is way better.
    The TrainingEnvironment was also used in Gupta et al. (2010)
    """
    exp_score = np.exp(score/tau)
    prob1 = exp_score/np.sum(exp_score)
    if straight_bias:
        prob2 = np.array([0.5, 0.183, 0.183, 0.0665, 0.0665, 0.01])  # Geometric sequence with u0 = 0.5, 2 * sum(u_i, i=1..(len(action)-1)/2) = 0.5
    else:
        prob2 = prob1
    res = np.random.choice(len(score), p=0.5*prob1+0.5*prob2)
    #print(score, prob, res)
    return res

def ep_lindyn_mg(env, theta, F, b, nb_day, nb_ep_per_day, log=None):
    """
    Does a linear dyna variation from Sutton et al. (2012) with replay.

    env - The environment to use
    theta - The weight vector to compute V(Phi)
    F - The transition tables from Phi to Phi', one per action
    b - A reward matrix which gives for each values of phi and an action the
        expected reward. For instance, if the 32 place cell is at the center of
        the environment, and action 8 is "going south", then because it's
        forbidden to go south at the center of the enviroment, b[32][8] will
        converge to -10. b is somewhat the Q(s, a) matrix.
    nb_day - number of "days" before ending the training. Days can also be
             understood as the number of replay sessions.
    nb_ep_per_day - number of time to do the task before going into "sleep mode".
                    The task is done `nb_day` * `nb_ep_per_day` in total.
    replay_max - Number of experienced feature activations to replay before
                 waking up.
    log - A list in which every place cells activation is recorded, along with
          the position of the agent and the position of the goal. While the
          agent sleeps, only the feature which is reactivated is logged.

    See Sutton et al. 2012 Dyna-Style Planning with Linear Function
    Approximation and Prioritized Sweeping for details about the algorithm (it
    is the algorithm 3 in the article). I have only moved the replay part so
    that it is not done after each step.
    """
    for day in range(nb_day):
        pqueue = PriorityQueue()
        for episode in range(nb_ep_per_day):
            if log is not None:
                log.append("session_begin")
            env.reinit()
            while not env.end:
                phi = env.get_features()
                q = np.array([-np.inf for i in env.action]) # Q of Q-learning
                for a in env.possible_actions():  # The impossible actions stay to -inf
                    q[a] = np.inner(b[a], phi) + gamma * np.dot(np.dot(theta.T, F[a]), phi)
                a = softmax(q, 2, straight_bias=False)
                phi_n, r = env.do_action(a)

                delta = r + gamma * np.dot(theta.T, phi_n) - np.dot(theta.T, phi)
                theta = theta + alpha * delta * phi
                F[a] = F[a] + np.outer(alpha * (phi_n - np.dot(F[a], phi)), phi)
                b[a] = b[a] + alpha * (r - np.inner(b[a], phi)) * phi
                for i in range(len(phi)):
                    if phi[i] != 0:
                        pqueue.put((-np.abs(delta * phi[i]), i))

                if log is not None:
                    log.append([env.get_features(), np.copy(env.pos), np.copy(env.goals[0])])
                    log.append("sleep")
                # Replay
                p = env.p # Number of replay max
                while not pqueue.empty() and p > 0:
                    unused_prio, i = pqueue.get()
                    if log is not None:
                        activation = np.zeros(env.pc.nb_place_cells)
                        activation[i] = 1
                        log.append([activation, None, None])
                    for j in range(F.shape[2]):
                        if np.any(F[:, i, j] != 0):
                            delta = -np.inf
                            for a in range(len(env.action)):
                                delta = np.max((delta, b[a][j] + gamma * np.dot(theta.T, F[a, :, j]))) - theta[j]
                            theta[j] = theta[j] + alpha * delta
                            pqueue.put((-np.abs(delta), j))
                    p -= 1
                if log is not None:
                    log.append("end")
    return theta, b, F
