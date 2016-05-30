import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import time
import sys
import pickle
import gzip


from environment import Environment, TrainingEnvironment, PlaceCells, TaskEnvironment
from linear_dyna import ep_lindyn_mg

def main():
    pc = PlaceCells(200, 100, 2000, 1000)

    print("entrainement gauche")
    env = TrainingEnvironment('L', 20, pc)
    theta = np.zeros(pc.nb_place_cells)
    F = np.zeros((len(env.action), pc.nb_place_cells, pc.nb_place_cells))
    b = np.zeros((len(env.action), pc.nb_place_cells))
    log = []
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 7, 1, 300)

    print("entrainement droite")
    env = TrainingEnvironment('R', 20, pc)
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 7, 1, 300)

    print("tache")
    env = TaskEnvironment('L', 10, pc)
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 1, 1, 300, log)
    env = TaskEnvironment('R', 10, pc)
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 1, 1, 300, log)
    env = TaskEnvironment('L', 10, pc)
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 1, 1, 300, log)
    print("fin")
    with gzip.open('log_{}.pklz'.format(int(time.time())), 'wb') as f:
        pickle.dump({'log': log, 'theta':theta, 'b': b, 'F': F, 'env': env}, f)
    intersting = []
    sleep = False
    for elem in log:
        if isinstance(elem, str) and elem == "end":
            sleep = False
        if sleep:
            intersting.append(elem)
        if isinstance(elem, str) and elem == "sleep":
            sleep = True
    animate(env, iter(log))

def animate(env, ep):
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

def play_ep():
    with gzip.open('log_{}.pklz'.format(1464356873), 'rb') as f:
        a = pickle.load(f)
    log = a['log']
    env = a['env']
    env.patches = env.init_patches()
    intersting = []
    sleep = False
    for elem in log:
        if isinstance(elem, str) and elem == "end":
            sleep = False
        if sleep:
            intersting.append(elem)
        if isinstance(elem, str) and elem == "sleep":
            sleep = True
    animate(env, iter(log))

main()
