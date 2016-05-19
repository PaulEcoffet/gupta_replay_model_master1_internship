import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import time
import sys


from environment import Environment
from linear_dyna import ep_lindyn_mg

def main():
    env = Environment(100, 100)
    theta = np.zeros(env.nb_place_cells)
    F = np.zeros((len(env.action), env.nb_place_cells, env.nb_place_cells))
    b = np.ones((len(env.action), env.nb_place_cells))
    episode = ep_lindyn_mg(env, theta, F, b, 5, 1, 400, True)
    animate(env, episode)

def animate(env, ep):
    prev_res = None

    def true_animate(value):
        try:
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
