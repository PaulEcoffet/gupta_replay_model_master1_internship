from __future__ import print_function

import numpy as np
import gzip
import argparse
import cPickle

def scatter(env, log, outfile):
    print("step", "cell", "x", "y", "activation", "replay", file=outfile)
    replay = False
    for step, e in enumerate(log):
        if isinstance(e, str):
            if e == "sleep":
                replay = True
            elif e == "end":
                replay = False
        else:
            cells = e[0]
            for i in np.where(cells > 0)[0]:
                print(step, i, env.pc.kernels[i][0], env.pc.kernels[i][1], cells[i], replay, file=outfile)

action = {'scatter': scatter}

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--replayonly', action='store_true')
parser.add_argument('action', choices=['scatter'])
parser.add_argument('infile', nargs='?', type=str,
                    default=None)
parser.add_argument('outfile', type=argparse.FileType('w'))
args = parser.parse_args()

with gzip.open(args.infile, "rb") as f:
    a = cPickle.load(f)
    log = a['log']
    env = a['env']
    action[args.action](env, log, args.outfile)
args.outfile.close()
