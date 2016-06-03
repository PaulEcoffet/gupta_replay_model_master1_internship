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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import time
import sys
import cPickle as pickle
import gzip
import sys


from environment import Environment, TrainingEnvironment, PlaceCells, TaskEnvironment
from linear_dyna import ep_lindyn_mg

def main(replay_only=False):
    pc = PlaceCells(100, 140, 2000, 1000)

    print("entrainement gauche (20 boucles par jour sur 7 jours)")
    env = TrainingEnvironment('L', 20, pc)
    theta = np.zeros(pc.nb_place_cells)
    F = np.zeros((len(env.action), pc.nb_place_cells, pc.nb_place_cells))
    b = np.zeros((len(env.action), pc.nb_place_cells))
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 7, 1, 300) # 7: nb jour, 1: nb de fois la tache (1 fois mais avec 20 tours)

    print("entrainement droite (20 boucles par jour sur 7 jours)")
    env = TrainingEnvironment('R', 20, pc)
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 7, 1, 300)

    log = []
    print("tache")
    env = TaskEnvironment('L', 10, pc) # 10 tours à gauche
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 1, 1, 300, log)
    env = TaskEnvironment('R', 10, pc) # 10 tours à droite
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 1, 1, 300, log)
    env = TaskEnvironment('L', 10, pc) # 10 tours à gauche
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 1, 1, 300, log)
    print("fin")

    # Sauvegarde du log (seulement partie tache) dans un fichier compressé
    path = 'log_{}.pklz'.format(int(time.time()))
    with gzip.open(path, 'wb') as f:
        pickle.dump({'log': log, 'theta':theta, 'b': b, 'F': F, 'env': env}, f)
    play_ep(path, replay_only)

def animate(env, ep):  # Some very weird things with matplotlib. I don't understand it either
    def true_animate(value):
        try:
            info = next(ep)
            while isinstance(info, str):
                print(info)
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
                                     blit=False, repeat=False, interval=16)
    if sys.argv[0] == "vid":
        print("video")
        im_ani.save('vids/replay_1_per_day_5_days{}.mp4'.format(int(time.time())),
                    codec="libx264", writer=animation.FFMpegFileWriter(),
                    fps=30, bitrate=1000,
                    extra_args=['-pix_fmt', 'yuv420p', '--verbose-debug'])
    else:
        plt.show()

def play_ep(path, replay_only=False): # Replay an episode from a log (the name must be put manually)
    with gzip.open(path, 'rb') as f:
        a = pickle.load(f)
    log = a['log']
    env = a['env']
    env.patches = env.init_patches()
    if replay_only:
        interesting = []
        sleep = False
        for elem in log:
            if isinstance(elem, str) and elem == "end":
                sleep = False
            if sleep:
                interesting.append(elem)
            if isinstance(elem, str) and elem == "sleep":
                sleep = True
    else:
        interesting = log
    animate(env, iter(interesting))


if __name__ == "__main__":
    arg = sys.argv
    if '-r' in arg:
        replay_only = True
    path = None
    for p in arg[1:]:
        if p != '-r':
            path = p
    if path is not None:
        play_ep(path, replay_only)
    else:
        main(replay_only)  # Change to play_ep() to replay a log
