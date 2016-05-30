from __future__ import division, print_function

import copy
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from matplotlib.collections import Collection, PatchCollection


class PlaceCells(object):
    def __init__(self, nb_place_cells, place_field_radius, W, H, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.nb_place_cells = nb_place_cells
        row = int(np.sqrt(nb_place_cells))
        self.kernels = np.array([((i%row) * W/row + np.random.randint(W/(row)),
                                  (i//row) * H / row + np.random.randint(H/(row)))
                        for i in range(nb_place_cells)])
        self.place_field_radius = place_field_radius
        self.place_field_radius = np.random.choice([4*place_field_radius/5, place_field_radius, 3*place_field_radius/2], nb_place_cells)
        self.sigma2 = self.place_field_radius**2

class Environment(object):
    def __init__(self, place_cells, seed=None):
        self.H = 1000
        self.W = 2000

        # Init definition
        self.pc = place_cells
        self.start = np.array((int(0.5 * self.W), int(0.2 * self.H)), dtype=float)
        self.goals_init = [np.array((int(0.9 * self.W), int(0.9 * self.H)), dtype=float)]
        self.goals = copy.deepcopy(self.goals_init)
        self.pos = np.copy(self.start)
        self.walls = []
        self.reward = 0
        self.end = False

        # action definition
        self.velocity = 70
        self.direction = np.pi / 2
        self.action = [i*np.pi/4 for i in range(8)]

        # Matplotlib bound variables
        d_dir = 50 * np.array([np.cos(self.direction), np.sin(self.direction)])
        self.pos_d_dir = self.pos + d_dir
        self.visual_pos = np.copy(self.pos)
        self.visual_goal = np.copy(self.goals[0])
        self.tr_orientation = self.direction - np.pi
        self.patches = self.init_patches()

    def reinit(self):
        self.pos = np.copy(self.start)
        d_dir = 50*np.array([np.cos(self.direction), np.sin(self.direction)])
        self.pos_d_dir = self.pos + d_dir
        self.goals = copy.copy(self.goals_init)
        self.direction = np.pi/2
        self.reward = 0
        self.end = False

    def init_patches(self):
        patches = OrderedDict({(x, y):
                    plt.Circle([x, y], self.pc.place_field_radius[i], fill=True,
                               fc=[1, 1, 1, 0], ec=[0, 0, 0, 0.4])
                    for i, (x, y) in enumerate(self.pc.kernels)})
        for i, wall in enumerate(self.walls):
            patches['wall', i] = mpatches.Rectangle((wall[0], wall[1]), wall[2] - wall[0], wall[3] - wall[1], fc=[0, 0, 0])
        patches['dir'] = mpatches.RegularPolygon(self.pos_d_dir, 3, orientation=self.tr_orientation, radius=30)
        patches['pos'] = mpatches.Circle(self.visual_pos, 50)
        patches['goal'] = mpatches.RegularPolygon(self.visual_goal, 5, radius=100, fc=(1, 0, 0))
        return patches

    def get_features(self):
        features = np.exp(-np.sum(np.power(self.pos - self.pc.kernels, 2), axis=1)/(self.pc.sigma2))
        features[np.linalg.norm(self.pos - self.pc.kernels, axis=1) > self.pc.place_field_radius] = 0
        #head_cells_kernels = np.linspace(-np.pi, np.pi, 10)
        #head_cells_features = np.exp(-np.power(self.direction - head_cells_kernels, 2)/0.5)
        #np.concatenate([features, head_cells_features], axis=0)
        return features

    def get_repr(self, features=None, pos=None, goal=None):
        if features is None:
            features = self.get_features()
        if pos is None:
            pos = self.pos
        if goal is None:
            goal = self.goals[0]
        for i, (x, y) in enumerate(self.pc.kernels):
            self.patches[x, y].set_fc([0.5, 0.5, 1, features[i]*3/4])
        d_dir = 50*np.array([np.cos(self.direction), np.sin(self.direction)])
        self.tr_orientation += self.direction - np.pi - self.tr_orientation
        self.pos_d_dir[0], self.pos_d_dir[1] = pos + d_dir # Prevent weird behaviour when id change for matplotlib, pos_d_dir is bound to the patch.
        self.visual_pos[0], self.visual_pos[1] = pos
        self.visual_goal[0], self.visual_goal[1] = goal
        collec = PatchCollection(self.patches.values(), match_original=True)
        return collec

    def do_action(self, action):
        new_pos, new_dir = self.sim_action(action)
        np.clip(new_pos, [0, 0], [self.W, self.H], out=new_pos)
        self.trigger_reward(self.pos, new_pos, action)
        if self.in_walls(new_pos):
            new_pos = np.copy(self.pos)
        self.pos[0] = new_pos[0]  # Prevent weird behaviour when id change for matplotlib, pos is bound to the patch.
        self.pos[1] = new_pos[1]
        return self.get_features(), copy.copy(self.reward)

    def trigger_reward(self, prev_pos, new_pos, a):
        for i, goal in enumerate(self.goals):
            if np.linalg.norm(new_pos - goal) < 140:
                print("yummi")
                self.goals[i] = np.array([-1000, -1000])
                self.end = True
                self.reward = 1
            self.reward = 0

    def possible_actions(self, pos=None, direction=None):
        possible = []
        for a in range(len(self.action)):
            new_pos, direc = self.sim_action(a)
            new_pos += self.velocity/2 * np.array([np.cos(direc), np.sin(direc)])
            if not (self.in_walls(new_pos) or self.out_of_bound(new_pos)):
                possible.append(a)
        return possible

    def sim_action(self, a, pos=None, direction=None):
        if pos is None:
            pos = self.pos
        if direction is None:
            direction = self.direction
        new_dir = self.action[a]
        new_pos = pos + np.array([self.velocity * np.cos(new_dir),
                                  self.velocity * np.sin(new_dir)])
        return new_pos, new_dir

    def in_walls(self, pos=None):
        if pos is None:
            pos = self.pos
        for wall in self.walls:
            if wall[0] < pos[0] < wall[2] and wall[3] < pos[1] < wall[1]:
                return True
        return False

    def out_of_bound(self, pos=None):
        if pos is None:
            pos = self.pos
        return not (0 < pos[0] < self.W and 0 < pos[1] < self.H)

    def on_start(self, pos=None):
        if pos is None:
            pos = self.pos
        return np.linalg.norm(pos - self.start) < 100

    def in_rect(self, pos, ix1, ix2, iy1, iy2):
        x1 = min(ix1, ix2)
        x2 = max(ix1, ix2)
        y1 = min(iy1, iy2)
        y2 = max(iy1, iy2)
        return x1*self.W <= pos[0] <= x2*self.W and y1*self.H <= pos[1] <= y2*self.H

class TrainingEnvironment(Environment):

    def __init__(self, side, max_turn, place_cells, seed=None):
        super(self.__class__, self).__init__(place_cells)
        if side == "R":
            self.walls = [[int(0.1 * self.W), int(1.0 * self.H), int(0.45 * self.W), int(0.0 * self.H)],
                          [int(0.55 * self.W), int(0.8 * self.H), int(0.9 * self.W), int(0.2 * self.H)]]
            self.goals_init = [np.array([0.9*self.W, 0.9*self.H])]
        else:
            self.walls = [[int(0.1 * self.W), int(0.8 * self.H), int(0.45 * self.W), int(0.2 * self.H)],
                          [int(0.55 * self.W), int(1.0 * self.H), int(0.8 * self.W), int(0.0 * self.H)]]
            self.goals_init = [np.array([0.1*self.W, 0.9*self.H])]
        self.patches = super(self.__class__, self).init_patches()
        self.nb_exp = 0
        self.max_turn = max_turn

    def trigger_reward(self, prev_pos, new_pos, a):
        self.reward = 0
        if self.in_rect(prev_pos, 0.45, 0.55, 0.0, 0.8):  ## Punish if going backward
            good_dir = np.array([0, 1])
        elif self.in_rect(prev_pos, 0.1, 0.45, 0.8, 1.0) or self.in_rect(prev_pos, 0.55, 1, 0.0, 0.2):
            good_dir = np.array([-1, 0])
        elif self.in_rect(prev_pos, 0.55, 0.9, 0.8, 1.0) or self.in_rect(prev_pos, 0, 0.45, 0.0, 0.2):
            good_dir = np.array([1, 0])
        elif self.in_rect(prev_pos, 0.9, 1, 0.2, 1) or self.in_rect(prev_pos, 0.0, 0.1, 0.2, 1):
            good_dir = np.array([0, -1])
        else:
            good_dir = None
        if good_dir is not None and np.dot(new_pos - prev_pos, good_dir) < 0:
            self.reward = -10
            new_pos[0] = prev_pos[0]
            new_pos[1] = prev_pos[1]
        if self.on_start(self.pos):
            self.goals = copy.deepcopy(self.goals_init)
        for i, goal in enumerate(self.goals):
            if np.linalg.norm(new_pos - goal) < 140:
                print("yummi")
                self.goals[i] = np.array([-1000, -1000])
                self.nb_exp += 1
                if self.nb_exp == self.max_turn:
                    self.nb_exp = 0
                    self.end = True


class TaskEnvironment(Environment):
    def __init__(self, side, max_turn, place_cells, seed=None):
        super(self.__class__, self).__init__(place_cells)
        self.walls = [[int(0.1 * self.W), int(0.8 * self.H), int(0.45 * self.W), int(0.2 * self.H)],
                      [int(0.55 * self.W), int(0.8 * self.H), int(0.9 * self.W), int(0.2 * self.H)]]
        if side == 'R':
            self.goals_init = [np.array([0.9*self.W, 0.9*self.H])]
        else:
            self.goals_init = [np.array([0.1*self.W, 0.9*self.H])]
        self.patches = super(self.__class__, self).init_patches()
        self.nb_exp = 0
        self.max_turn = max_turn

    def trigger_reward(self, prev_pos, new_pos, a):
        ### KEEP THE PUNITION IF BACKWARD TO PREVENT SURPRISE
        self.reward = 0
        if self.in_rect(prev_pos, 0.45, 0.55, 0.0, 0.8):  ## Punish if going backward
            good_dir = np.array([0, 1])
        elif self.in_rect(prev_pos, 0.1, 0.45, 0.8, 1.0) or self.in_rect(prev_pos, 0.55, 1, 0.0, 0.2):
            good_dir = np.array([-1, 0])
        elif self.in_rect(prev_pos, 0.55, 0.9, 0.8, 1.0) or self.in_rect(prev_pos, 0, 0.45, 0.0, 0.2):
            good_dir = np.array([1, 0])
        elif self.in_rect(prev_pos, 0.9, 1, 0.2, 1) or self.in_rect(prev_pos, 0.0, 0.1, 0.2, 1):
            good_dir = np.array([0, -1])
        else:
            good_dir = None
        if good_dir is not None and np.dot(new_pos - prev_pos, good_dir) < 0:
            self.reward = -10
            new_pos[0] = prev_pos[0]
            new_pos[1] = prev_pos[1]
        if self.on_start(self.pos):
            self.goals = copy.deepcopy(self.goals_init)
        for i, goal in enumerate(self.goals):
            if np.linalg.norm(new_pos - goal) < 140:
                print("yummi")
                self.goals[i] = np.array([-1000, -1000])
                self.nb_exp += 1
                if self.nb_exp == self.max_turn:
                    self.nb_exp = 0
                    self.end = True
                self.reward = 100

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
