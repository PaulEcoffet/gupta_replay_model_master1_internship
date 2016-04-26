from __future__ import division


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from matplotlib.collections import PatchCollection



class Environment:

    def __init__(self, nb_place_cells, place_field_radius, seed=None):
        self.H = 1000
        self.W = 1000
        self.kernels = np.array([(np.random.randint(self.H),
                                  np.random.randint(self.W))
                        for dummy in range(nb_place_cells)])
        self.place_field_radius = place_field_radius
        self.sigma2 = 10000
        self.eps = np.exp(-self.place_field_radius**2 / (2*self.sigma2))
        print("eps", self.eps)
        self.start = np.array((int(0.5 * self.W), int(0.2 * self.H)))
        self.goal = np.array((int(0.8 * self.W), int(0.8 * self.H)))
        self.pos = self.start

    def get_features(self):
        features = np.exp(-np.sum(np.power(self.pos - self.kernels, 2), axis=1)/(2*self.sigma2))
        features[features <= self.eps] = 0
        return features

    def get_repr(self):
        features = self.get_features()
        patches = [
            plt.Circle([x, y], self.place_field_radius, fill=True,
                       fc=[0.5, 0.5, 1, features[i]/2], ec=[0, 0, 0, 0.5])
            for i, (x, y) in enumerate(self.kernels)]
        patches.append(mpatches.RegularPolygon(self.pos, 5, radius=10))
        collec = PatchCollection(patches, match_original=True)
        return patches

def test():
    env = Environment(75, 150)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim([0, env.H])
    ax.set_xlim([0, env.W])
    frames = []
    for i in range(10):
        frames.append(env.get_repr())
        env.pos[1] += 120
    im_ani = animation.FuncAnimation(fig, animate, frames=frames, repeat=False, interval=50, blit=False)

    plt.show()

def animate(frame):
    print(frame)
    return [frame, ]
