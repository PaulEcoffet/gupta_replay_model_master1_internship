from __future__ import print_function

"""
Export data from the log in .pklz in .csv
USAGE
```
export_to_csv.py activation log.pklz out.csv
```
"""

import numpy as np
import gzip
import argparse
import cPickle

def activation(env, log, outfile):
    """
    Export which cells are activated at each time step and log if it is a replay
    or not.
    """
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


action = {'activation': activation}

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--replayonly', action='store_true') # Not implemented
parser.add_argument('action', choices=['activation'])
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
