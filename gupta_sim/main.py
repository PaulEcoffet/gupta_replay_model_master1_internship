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
    pc = PlaceCells(100, 100, 1000, 1000)
    env = TrainingEnvironment('L', 20, pc)
    theta = np.zeros(pc.nb_place_cells)
    F = np.zeros((len(env.action), pc.nb_place_cells, pc.nb_place_cells))
    b = np.zeros((len(env.action), pc.nb_place_cells))
    log = []
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 2, 1, 300)
    env = TrainingEnvironment('R', 20, pc)
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 2, 1, 300)
    env = TaskEnvironment('R', 10, pc)
    theta, b, F = ep_lindyn_mg(env, theta, F, b, 3, 1, 300, log)
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
    print(len(intersting))
    animate(env, iter(intersting))

def animate(env, ep):
    prev_res = None

    def true_animate(value):
        try:
            features = next(ep)
            while isinstance(features, str):
                print(features)
                features = next(ep)
        except StopIteration:
            pass
        else:
            ax.clear()
            res = ax.add_collection(env.get_repr(features))
        return [],

    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca()
    ax.set_ylim([0, env.H])
    ax.set_xlim([0, env.W])

    im_ani = animation.FuncAnimation(fig, true_animate, frames=16000,
                                     blit=False, repeat=False, interval=30)
    if sys.argv[0] == "vid":
        print("video")
        im_ani.save('vids/replay_1_per_day_5_days{}.mp4'.format(int(time.time())),
                    codec="libx264", writer=animation.FFMpegFileWriter(),
                    fps=30, bitrate=1000,
                    extra_args=['-pix_fmt', 'yuv420p', '--verbose-debug'])
    else:
        plt.show()
    plt.close()

main()
