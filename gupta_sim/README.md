Read me
=======

To run the program, the tkinter backend of matplotlib must be used. Other backends do not handle animations well. To do so, you have to run these commands:

```bash
$ ipython
In[1]: %matplotlib tk
In[2]: %run main
```

Please note that in the code, the `print` function and the `division` are the ones from python 3. It means that `/` returns a float and `//` return an euclidian division.
`print` is a function and not a keyword, parenthesis must be added, it can be passed as an argument and has the syntax `print(ma_var, file=f)` which is really nice.


The environment has to have a lot of specific properties so that linear dyna works. Read carefull these conditions of the Sutton et al. paper.
Overlapping features is one of the reason I think linear dyna has convergence issue, yet it is also what make the hypothesis of surprisal possible.

Interesting readings to understand this code:

Gupta, A. S., van der Meer, M. A. A., Touretzky, D. S., \& Redish, A. D. (2010). Hippocampal Replay Is Not a Simple Function of Experience. Neuron, 65(5), 695‑705. http://doi.org/10.1016/j.neuron.2010.01.034

Sutton, R. S., Szepesvári, C., Geramifard, A., \& Bowling, M. P. (2012). Dyna-style planning with linear function approximation and prioritized sweeping. arXiv preprint arXiv:1206.3285. Consulté à l’adresse http://arxiv.org/abs/1206.3285

Vanseijen, H., \& Sutton, R. (2015). A Deeper Look at Planning as Learning from Replay. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15) (p. 2314–2322). Consulté à l’adresse http://machinelearning.wustl.edu/mlpapers/paper_files/icml2015_vanseijen15.pdf
