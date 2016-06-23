# Algorithmes

[ ] Vérifier l'implémentation du Linear Dyna qui contient peut être des fautes d'implémentation
[ ] Essayer de comprendre pourquoi les delta sont si petits si rapidement et savoir si c'est un défaut
[ ]
[ ] Essayer d'utiliser du prioritized sweeping avec un Forgetful LSTD présenté par Van Seijen \& Sutton (2015) et en vérifier la convergence.
  [ ] Essayer sur un monde à case simple ?

# Environment

[ ] Reproduire exactement l'environment issu de Gupta et al. (avec dimensions correctes)
  [ ] Revoir le système de mur et de direction forcée
[ ] Choisir si direction forcée ou entrainement à suivre une direction (comme dans Gupta)
[ ] Choisir si bruit de récompense ou bruit de paramètres par défaut
[ ] Choisir nombre de cellule de lieu et taille du champ récepteur
  [ ] Regarder cela dans la littérature

# Analyse de données

[ ] Pouvoir déterminer si une séquence est backward, forward, sa position dans le labyrinthe
  [ ] Utiliser la même détection de séquences que Gupta ?
  [ ] Trouver à quel point il est pertinent de moyenner sur des séquences de replays
[ ]
