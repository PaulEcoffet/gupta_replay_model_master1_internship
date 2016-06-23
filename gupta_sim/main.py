# -*- coding: utf8 -*-
"""
Reproduce a simplified version of Gupta et al (2010) experiment. To run it you
need to do this exactly (the animation does not work if you don't do that):

You need to force the tk backend !

In bash:
```
Documents/stage/gupta_sim$ ipython
Python 2.7.11+ (default, Apr 17 2016, 14:00:29)
Type "copyright", "credits" or "license" for more information.

IPython 4.2.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: # Animation works only with tk backend

In [2]: %matplotlib tk

In [3]: %run main
```
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import time
import argparse
import sys
import cPickle as pickle
import gzip


from environment import Environment, TrainingEnvironment, PlaceCells, TaskEnvironment
from linear_dyna import ep_lindyn_mg

np.set_printoptions(threshold=np.nan, precision=2)

def main(replayonly=False):

    pc = PlaceCells(100, 140, 2000, 1000)
    env = TrainingEnvironment('L', 20, pc)
    theta = np.zeros(pc.nb_place_cells)
    F = np.zeros((len(env.action), pc.nb_place_cells, pc.nb_place_cells))
    b = np.zeros((len(env.action), pc.nb_place_cells))

    try:
        log = []
        print("tache")
        env = TaskEnvironment('L', 10, pc) # 10 tours à gauche
        theta, b, F, pqueue, step = ep_lindyn_mg(env, theta, F, b, 1, 1, None, log=log)
        env = TaskEnvironment('R', 10, pc) # 10 tours à droite
        theta, b, F, pqueue, step = ep_lindyn_mg(env, theta, F, b, 1, 1, pqueue, step, log)
        env = TaskEnvironment('L', 10, pc) # 10 tours à gauche
        theta, b, F, pqueue, step = ep_lindyn_mg(env, theta, F, b, 1, 1, pqueue, step, log)
        print("fin")
    finally:
        pass

    # Plot de la value function avec une échelle logarithmique.
    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca()
    ax.set_ylim([0, env.H])
    ax.set_xlim([0, env.W])
    logt = np.log(theta - np.min(theta) + 1)
    print(logt)
    ax.add_collection(env.get_repr((logt - np.min(logt))/((np.max(logt) - np.min(logt))), env.pos, (0, 0)))
    plt.show()
    raw_input("next? ")

    # Sauvegarde du log dans un fichier compressé
    filename = 'log_{}.pklz'.format(int(time.time()))
    with gzip.open(filename, 'wb') as f:
        pickle.dump({'log': log, 'theta':theta, 'b': b, 'F': F, 'env': env}, f)
    play_ep(filename, replayonly)


def animate(env, ep):  # Some very weird things with matplotlib. I don't understand it either
    def true_animate(value):
        try:
            info = next(ep)
            while isinstance(info, str):
                info = next(ep)
        except StopIteration:
            pass
        else:
            features = info[0]
            pos = info[1]
            goal = info[2]
            ax.clear()
            ax.add_collection(env.get_repr(features, pos, goal))
        return [],

    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca()
    ax.set_ylim([0, env.H])
    ax.set_xlim([0, env.W])
    im_ani = animation.FuncAnimation(fig, true_animate, frames=16000,
                                     blit=False, repeat=False, interval=2)
    if False: # Video encoding is not that great
        print("video")
        im_ani.save('vids/{}.mp4'.format(int(time.time())),
                    codec="libx264", writer=animation.FFMpegFileWriter(),
                    fps=30, bitrate=1000,
                    extra_args=['-pix_fmt', 'yuv420p', '--verbose-debug'])
    else:
        plt.show()


def play_ep(file_, replayonly=False): # Replay an episode from a log (the name must be put manually)
    with gzip.open(file_, 'rb') as f:
        a = pickle.load(f)
    log = a['log']
    env = a['env']
    env.patches = env.init_patches()
    if replayonly:
        interesting = []
        sleep = False
        for elem in log:
            if isinstance(elem, str) and elem == "end":
                sleep = False
            if sleep:
                interesting.append(elem)
            if isinstance(elem, str) and elem == "sleep":
                sleep = True
        animate(env, iter(interesting))
    else:
        animate(env, iter(log))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--replayonly', action='store_true')

    parser.add_argument('infile', nargs='?', type=str,
                        default=None)
    args = parser.parse_args()
    elif args.infile is None:
        main(args.replayonly)
    else:
        play_ep(args.infile, args.replayonly)
