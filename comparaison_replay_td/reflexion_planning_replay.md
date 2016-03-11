% Notes sur replay, model free, model based
% Paul Ecoffet

En quelque sorte, replay est apprentissage par regret : Si j'avais su ceci à
l'époque, alors j'aurais pu prévoir cela.

Pourquoi est-ce aussi du planning ?

Le planning, c'est construire un modèle de son environnement. En fait, à partir
d'expériences passées, on a une estimation des gains des états et des
transitions des états. Planifier, c'est donc estimer le chemin qui nous fera
maximiser les gains en prenant en compte les proba de transition d'un état vers
un autre. Un état qui apporte un fort gain mais qui mène nécessairement vers
une forte punition doit ainsi être évité.

C'est bien *notre expérience passée* qui construit le modèle => lien fort entre
replay et planning

Différence entre lina dyna et algo van Seijen, c'est que van Seijen ne tire pas
les expériences des états probables mais des états déjà rencontrés uniquement
Pertinent ?  Oui, les états réguliers seront ajustés le plus, les moins régulier
seront ajusté moins souvent. La distrib empirique construite aurait le même
comportement

# Est-ce que model-free $\Leftrightarrow$ model-based?

Un algo model-free ne construit pas de représentation de son environnement.
Cependant, il est difficile de trancher pour savoir ce qui est une
représentation de l'environnement et ce qui ne l'est pas. Par exemple, parmi ces
données, lesquelles sont sans modèle et lesquelles sont avec modèle ?

* $\pi(s)$, la politique de l'agent
* $V_{agent}$, les valeurs des états inférées par l'agent
* $e(s)$, la trace d'éligibilité d'un état
* $(S_t, A_t)$, l'ensemble des actions prises par l'agent, dans l'ordre
* $(S_m, A_m)$, les $m$ dernières actions prises par l'agent, dans l'ordre
* $F_\pi$, la matrice de transition d'un état à l'autre, inférée par l'agent

# Pourquoi le *Forgetful LSTD($\lambda$)* est model-based ?
