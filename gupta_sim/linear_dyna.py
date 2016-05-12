import numpy as np
from Queue import PriorityQueue

INFTY = float("inf")

gamma = 0.9
alpha = 0.1

def softmax(score, tau):
    exp_score = np.exp(score/tau)
    prob = exp_score/np.sum(exp_score)
    res = np.random.choice(len(score), p=prob)
    return res

def ep_lindyn_mg(env, theta, F, b, nb_day, nb_ep_per_day, replay_max, do_yield=False):
    for day in range(nb_day):
        pqueue = PriorityQueue()
        for episode in range(nb_ep_per_day):
            env.reinit()
            while not env.end:
                phi = env.get_features()
                q = np.array([0. for i in env.action]) # Q of Q-learning
                for a in range(len(q)):
                    q[a] = np.inner(b[a], phi) + gamma * np.dot(np.dot(theta.T, F[a]), phi)
                a = softmax(q, 1)
                phi_n, r = env.do_action(a)

                delta = r + gamma * np.dot(theta.T, phi_n) - np.dot(theta.T, phi)
                theta = theta + alpha * delta * phi
                F[a] = F[a] + np.outer(alpha * (phi_n - np.dot(F[a], phi)), phi)
                b[a] = b[a] + alpha * (r - np.inner(b[a], phi)) * phi
                for i in range(len(phi)):
                    if phi[i] != 0:
                        pqueue.put((-np.abs(delta * phi[i]), i))
                if do_yield:
                    yield env.get_features()
        # end of episode
        #print("sleeping")
        # Sleep
        p = replay_max # Number of replay max
        while not pqueue.empty() and p > 0:
            unused_prio, i = pqueue.get()
            if do_yield:
                yield np.array([0 if j != i else 1 for j in range(len(env.kernels))])
            for j in range(F.shape[2]):
                if np.any(F[:, i, j] != 0):
                    delta = -INFTY
                    for a in range(len(env.action)):
                        delta = np.max((delta, b[a][j] + gamma * np.dot(theta.T, F[a, :, j]))) - theta[j]
                    theta[j] = theta[j] + alpha * delta
                    pqueue.put((-np.abs(delta), j))
            p -= 1

        #print (theta)
        #print (F)
    #return theta, b, F
