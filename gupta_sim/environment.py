# -*- coding: utf8 -*-

from __future__ import division, print_function

import copy
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from matplotlib.collections import Collection, PatchCollection

# PLEASE NOTICE THAT I USE DIVISION AND PRINT FROM PYTHON 3


class PlaceCells(object):
    def __init__(self, nb_place_cells, place_field_radius, W, H, seed=None):
        """
        Spread `nb_place_cells` place cell receptive fields for an environment
        of width `W` and height `H`.
        The `place_field_radius` is the size of the medium sized receptive field.
        Each place cell has a radius of either `place_field_radius`, 4/5 of
        `place_field_radius` or 3/2 of `place_field_radius`.

        The place cells receptive fields are placed so that they are equidistant
        with some noise (+/- space_between two receptive field/2)

        The seed can be fixed with the `seed` argument, so that the same place
        cells are yield.

        Each place cells has a center (kernel) and fire according to a gaussian
        radial basis function (see
        https://en.wikipedia.org/wiki/Radial_basis_function_kernel). It is
        computed according to the location of the agent in
        `Environment.get_features`.
        """
        if seed is not None:
            np.random.seed(seed)
        self.nb_place_cells = nb_place_cells
        row = int(np.sqrt(nb_place_cells))
        self.kernels = np.array([((i%row) * W/row + np.random.randint(W/(row)),
                                  (i//row) * H / row + np.random.randint(H/(row)))
                        for i in range(nb_place_cells)])
        self.place_field_radius = place_field_radius
        self.place_field_radius = np.random.choice([4*place_field_radius/5, place_field_radius, 3*place_field_radius/2], nb_place_cells)
        self.sigma2 = 4*self.place_field_radius**2

class Environment(object):
    def __init__(self, place_cells, seed=None):
        """
        The Environment class is an abstract class to implement different types
        of environments (TrainingEnvironment and TaskEnvironment). In
        Environment, the size of the maze is defined, some default parameters
        are also set: the start position, the velocity of the agent, and so on.

        Some Matplotlib specific variables are also defined to overcome
        matplotlib weird behaviours.
        """
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
        self.default_p = 0
        self.p = self.default_p # number of replay per sequence

        # Matplotlib bound variables
        d_dir = 50 * np.array([np.cos(self.direction), np.sin(self.direction)])
        self.pos_d_dir = self.pos + d_dir
        self.visual_pos = np.copy(self.pos)
        self.visual_goal = np.copy(self.goals[0])
        self.tr_orientation = self.direction - np.pi
        self.patches = self.init_patches()

    def reinit(self):
        """
        Set the Environment back to its original state.
        """
        self.pos = np.copy(self.start)
        d_dir = 50*np.array([np.cos(self.direction), np.sin(self.direction)])
        self.pos_d_dir = self.pos + d_dir
        self.goals = copy.copy(self.goals_init)
        self.direction = np.pi/2
        self.reward = 0
        self.end = False

    def init_patches(self):
        """
        Init Matplotlib patches, which are used to represent receptive fields,
        the agent and the goal.
        """
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
        """
        Get the features vector from the place cells. Each i-th value of the
        list represents the activation of the i-th place cells. Each place cell
        fires according to the position of the agent in the Environment. If
        the agent is on the center of the cell receptive field, then the
        activation is maximum. The more distant the agent is from the center
        of the receptive field, the lesser is the activation, following a
        gaussian radial basis function
        (see https://en.wikipedia.org/wiki/Radial_basis_function_kernel). If
        the rat is out of the receptive field, then the activation is 0.
        """
        features = np.exp(-np.sum(np.power(self.pos - self.pc.kernels, 2), axis=1)/(self.pc.sigma2))
        features[np.linalg.norm(self.pos - self.pc.kernels, axis=1) > self.pc.place_field_radius] = 0
        np.concatenate([features, [self.out_of_bound(self.goals[0])]]) # The rat knows if he has eaten recently.
        #head_cells_kernels = np.linspace(-np.pi, np.pi, 10)
        #head_cells_features = np.exp(-np.power(self.direction - head_cells_kernels, 2)/0.5)
        #np.concatenate([features, head_cells_features], axis=0)
        return features

    def get_repr(self, features=None, pos=None, goal=None):
        """
        Update the matplotlib bound variables so that the plot is updated. It
        is a very messy function.
        """
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
        """
        Move the agent, compute the reward it gets. A new time step occurs each
        time `do_action` is called.

        action - the id of the action to do (0, 1, 2, ...; not np.pi, -np.pi/2)
        """
        new_pos, new_dir = self.sim_action(action)
        np.clip(new_pos, [0, 0], [self.W, self.H], out=new_pos)
        self.trigger_reward(self.pos, new_pos, action)
        if self.in_walls(new_pos):
            new_pos = np.copy(self.pos)
        self.pos[0] = new_pos[0]  # Prevent weird behaviour when id change for matplotlib, pos is bound to the patch.
        self.pos[1] = new_pos[1]
        return self.get_features(), copy.copy(self.reward)

    def trigger_reward(self, prev_pos, new_pos, a):
        """
        Compute the reward for the agent and move the goal out of the environment
        so that it cannot be consume consecutively. In theory, several goals
        can be placed in the environment, but it will likely not work yet.
        """
        self.p = self.default_p
        self.reward = 0

    def possible_actions(self, pos=None, direction=None):
        """
        Lists all the action the agent can do without going into a wall or out
        of the environment.
        """
        if pos is None:
            pos = self.pos
        if self.in_rect(pos, 0.45, 0.55, 0.0, 0.8):  ## Punish if going backward
            good_dir = np.array([0, 1])
        elif self.in_rect(pos, 0.1, 0.45, 0.8, 1.0) or self.in_rect(pos, 0.55, 1, 0.0, 0.2):
            good_dir = np.array([-1, 0])
        elif self.in_rect(pos, 0.55, 0.9, 0.8, 1.0) or self.in_rect(pos, 0, 0.45, 0.0, 0.2):
            good_dir = np.array([1, 0])
        elif self.in_rect(pos, 0.9, 1, 0.2, 1) or self.in_rect(pos, 0.0, 0.1, 0.2, 1):
            good_dir = np.array([0, -1])
        else:
            good_dir = np.array([0, 0])

        possible = []
        for a in range(len(self.action)):
            new_pos, direc = self.sim_action(a)
            new_pos += self.velocity/2 * np.array([np.cos(direc), np.sin(direc)])
            if not (self.in_walls(new_pos) or self.out_of_bound(new_pos)) and np.inner(new_pos - pos, good_dir) >= 0:
                possible.append(a)
        if len(possible) == 0:
            raise Exception("agent is completly stuck")
        return possible

    def sim_action(self, a, pos=None, direction=None):
        """
        Compute the next position of the agent if it does the action `a` from
        the pos `pos` and with the direction `direction` (only works if
        direction is egocentric, and not implemented yet). It does not take into
        account walls and out of bound.

        If pos is None, then the actual position of the agent is used.
        If direction is None, then the actual direction of the agent is used.
        """
        if pos is None:
            pos = self.pos
        if direction is None:
            direction = self.direction
        new_dir = self.action[a]
        new_pos = pos + np.array([self.velocity * np.cos(new_dir),
                                  self.velocity * np.sin(new_dir)])
        return new_pos, new_dir

    def in_walls(self, pos=None):
        """
        Say if the position `pos` is in a wall. If pos is ommited, then the
        actual position of the agent is used.
        """
        if pos is None:
            pos = self.pos
        for wall in self.walls:
            if wall[0] < pos[0] < wall[2] and wall[3] < pos[1] < wall[1]:
                return True
        return False

    def out_of_bound(self, pos=None):
        """
        Say if the position `pos` is out of bound. If pos is ommited, then the
        actual position of the agent is used.
        """
        if pos is None:
            pos = self.pos
        return not (0 < pos[0] < self.W and 0 < pos[1] < self.H)

    def on_start(self, pos=None):
        """
        Say if the position `pos` is at the starting postion.
        If pos is ommited, then the actual position of the agent is used.
        The starting position is used te place a new reward on the goal site.
        """
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
    """
    Define the two training environments inspired from Gupta et al. (2010).
    The training environment is an environment without any reward, but where
    the agent is forced to go in a specific direction. If it does not, the he
    receives a negative reward and does not move, which is the same protocol as
    Gupta et al. If the rat turn back, the experimenter catch it, and replace
    it at its previous position.

    There is two different environment which can be generated, the Left
    environment and the Right environment, which looks like this:

    Right:
        +-----------------------------------------------------------+
        |             |XXXXX|                                       |
        |             |XXXXX|     No     +----------------->   +    |
        |             |XXXXX| Constraint                       |    |
        |             |XXXXX|                                  |    |
        |             |XXXXX|           +------------------+   |    |
        |             |XXXXX|     ^     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     |XXXXXXXXXXXXXXXXXX|   |    |
        |             |XXXXX|     |     +------------------+   v    |
        |             |XXXXX|     |                                 |
        |             |XXXXX|     +    <------------------------+   |
        |             |XXXXX|                                       |
        +-----------------------------------------------------------+

    And Left is symmetrical to this.

    The task last `max_turn` turns, which can be defined in the __init__.

    """

    def __init__(self, side, max_turn, place_cells, seed=None):
        """
        Define the TrainingEnvironment, with:

        side        - 'L' or 'R', the side of the loop
        max_turn    - The number of turn to do before the end of the task
        place_cells - The place cells to use to get the features from the
                      environment
        """
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
        """
        Punish if the agent goes backward and prevent it from moving (with a
        ugly trick)
        """
        self.reward = 0
        self.p = self.default_p
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
            self.reward = 0
            new_pos[0] = prev_pos[0]
            new_pos[1] = prev_pos[1]
        if self.on_start(self.pos):
            self.goals = copy.deepcopy(self.goals_init)
        for i, goal in enumerate(self.goals):
            if np.linalg.norm(new_pos - goal) < 140:
                self.goals[i] = np.array([-1000, -1000])
                self.nb_exp += 1
                self.p = 20
                if self.nb_exp == self.max_turn:
                    self.nb_exp = 0
                    self.end = True


class TaskEnvironment(Environment):
    """
    The task environment is like the TrainingEnvironment but with both the loops
    open and two reward sites at the top corners.
    The reward can be either at the left corner or the right corner depending
    of the condition. The punition if the agent goes backward is kept to avoid
    a big difference between its estimate and the reality which can trigger
    unwanted replays.

        +-----------------------------------------------------------+
        |                                                           |
        |   +    <--------+       No     +----------------->   +    |
        | REWARD              Constraint                     REWARD |
        |   |                                                  |    |
        |   | +--------------+          +------------------+   |    |
        |   | |XXXXXXXXXXXXXX|    ^     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | |XXXXXXXXXXXXXX|    |     |XXXXXXXXXXXXXXXXXX|   |    |
        |   | +--------------+    |     +------------------+   v    |
        |   v                     |                                 |
        |    +--------------->    +    <------------------------+   |
        |                                                           |
        +-----------------------------------------------------------+

    """
    def __init__(self, side, max_turn, place_cells, seed=None):
        """
        Define the TaskEnvironment with:

        side - The side in which the reward will be, either 'L', 'R' or 'both'
               ('both' is not implemented yet)
        max_turn - The number of reward to gather to end the task
        place_cells - The place cells to use to compute the features vector.
        """
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
        """
        Gives the reward to the agent and put the reward out of reach until it
        goes to the start position and a new reward is then put.
        """
        ### KEEP THE PUNITION IF BACKWARD TO PREVENT SURPRISE
        self.reward = 0
        self.p = self.default_p
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
            self.reward = 0
            new_pos[0] = prev_pos[0]
            new_pos[1] = prev_pos[1]
        if self.on_start(self.pos):
            self.goals = copy.deepcopy(self.goals_init)
        for i, goal in enumerate(self.goals):
            if np.linalg.norm(new_pos - goal) < 140:
                self.goals[i] = np.array([-1000, -1000])
                self.nb_exp += 1
                if self.nb_exp == self.max_turn:
                    self.nb_exp = 0
                    self.end = True
                self.reward = 10
                self.p = 200
                break
        self.reward += np.random.normal(scale=0.01)
