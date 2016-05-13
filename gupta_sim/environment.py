from __future__ import division, print_function

import copy
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from matplotlib.collections import Collection, PatchCollection


class Environment:

    def __init__(self, nb_place_cells, place_field_radius, seed=None):
        self.H = 1000
        self.W = 1000
        if seed is not None:
            np.random.seed(seed)
        # Place Cells definition
        self.nb_place_cells = nb_place_cells
        row = int(np.sqrt(nb_place_cells))
        self.kernels = np.array([((i%row) * self.W/row + np.random.randint(self.W/(row)),
                                  (i//row) * self.H / row + np.random.randint(self.H/(row)))
                        for i in range(nb_place_cells)])
        self.place_field_radius = place_field_radius
        self.place_field_radius = np.random.choice([4*place_field_radius/5, place_field_radius, 3*place_field_radius/2], nb_place_cells)
        self.sigma2 = self.place_field_radius**2

        # Init definition
        self.start = np.array((int(0.5 * self.W), int(0.2 * self.H)), dtype=float)
        self.goal = np.array((int(0.9 * self.W), int(0.9 * self.H)), dtype=float)
        self.pos = np.copy(self.start)

        self.walls = [[int(0.2 * self.W), int(0.8 * self.H), int(0.4 * self.W), int(0.2 * self.H)],
                      [int(0.6 * self.W), int(0.8 * self.H), int(0.8 * self.W), int(0.2 * self.H)]]
        self.reward = 0
        self.end = False

        # action definition
        self.velocity = 100
        self.direction = np.pi / 2
        self.action = [0]
        for i in [6, 4]:
            for j in [1, -1]:
                if j*np.pi/i not in self.action:
                    self.action.append(j*np.pi/i)


        # Matplotlib bound variables
        d_dir = 50*np.array([np.cos(self.direction), np.sin(self.direction)])
        self.pos_d_dir = self.pos + d_dir
        self.tr_orientation = self.direction - np.pi
        self.patches = self.init_patches()



    def reinit(self):
        self.start = np.array((int(0.5 * self.W), int(0.2 * self.H)), dtype=float)
        self.goal = np.array((int(0.9 * self.W), int(0.9 * self.H)), dtype=float)
        self.reward = 0
        self.end = False


    def init_patches(self):
        patches = OrderedDict({(x, y):
                    plt.Circle([x, y], self.place_field_radius[i], fill=True,
                               fc=[1, 1, 1, 0], ec=[0, 0, 0, 0.4])
                    for i, (x, y) in enumerate(self.kernels)})
        for i, wall in enumerate(self.walls):
            patches['wall', i] = mpatches.Rectangle((wall[0], wall[1]), wall[2] - wall[0], wall[3] - wall[1], fc=[0, 0, 0])
        patches['dir'] = mpatches.RegularPolygon(self.pos_d_dir, 3, orientation=self.tr_orientation, radius=30)
        patches['pos'] = mpatches.Circle(self.pos, 50)
        patches['goal'] = mpatches.RegularPolygon(self.goal, 5, radius=100, fc=(1, 0, 0))
        return patches

    def get_features(self):
        features = np.exp(-np.sum(np.power(self.pos - self.kernels, 2), axis=1)/(self.sigma2))
        features[np.linalg.norm(self.pos - self.kernels, axis=1) > self.place_field_radius] = 0
        head_cells_kernels = np.linspace(-np.pi, np.pi, 10)
        head_cells_features = np.exp(-np.power(self.direction - head_cells_kernels, 2)/0.5)
        np.concatenate([features, head_cells_features], axis=0)
        return features

    def get_repr(self, features=None):
        if features is None:
            features = self.get_features()
        for i, (x, y) in enumerate(self.kernels):
            self.patches[x, y].set_fc([0.5, 0.5, 1, features[i]*3/4])
        d_dir = 50*np.array([np.cos(self.direction), np.sin(self.direction)])
        self.tr_orientation += self.direction - np.pi - self.tr_orientation
        self.pos_d_dir[0], self.pos_d_dir[1] = self.pos + d_dir # Prevent weird behaviour when id change for matplotlib, pos_d_dir is bound to the patch.
        collec = PatchCollection(self.patches.values(), match_original=True)
        return collec

    def do_action(self, action):
        self.direction += action
        noisy_dir = self.direction #+ np.random.normal(scale=np.pi/20)
        new_pos = self.pos + np.array([self.velocity * np.cos(noisy_dir),
                                       self.velocity * np.sin(noisy_dir)])

        if not (0 < new_pos[0] < self.W and 0 < new_pos[1] < self.H):
            np.clip(new_pos, [0, 0], [self.W, self.H], out=new_pos)
            self.reward = 0
        elif self.in_walls(new_pos):
            new_pos = np.copy(self.pos)
            self.reward = 0
        elif self.on_reward(new_pos):
            self.reward = 10
            self.end = True
            new_pos = np.copy(self.start)
        else:
            self.reward = 0

        self.pos[0] = new_pos[0]  # Prevent weird behaviour when id change for matplotlib, pos is bound to the patch.
        self.pos[1] = new_pos[1]

        return self.get_features(), copy.copy(self.reward)

    def in_walls(self, pos):
        for wall in self.walls:
            if wall[0] < pos[0] < wall[2] and wall[3] < pos[1] < wall[1]:
                return True
        return False

    def on_reward(self, pos):
        return np.linalg.norm(self.pos - self.goal) < 100

def animate(j, env, ax):
    res = ax.add_collection(env.get_repr())
    env.do_action(0)
    return res,


def test(time=100):
    #plt.ion()
    env = Environment(75, 140)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim([0, env.H])
    ax.set_xlim([0, env.W])

    #im_ani = animation.FuncAnimation(fig, animate, frames=time, fargs=(env, ax), repeat=True, interval=10, blit=True)
    im_ani.save("place_cells.mp4", dpi=500)
    plt.show(im_ani)
