# Hypothèses Body Pose — pré-enregistrées

Date de rédaction : 16 octobre 2026, avant toute exécution de `highline` sur les données Body Pose en mode complet.
Statut : **exploratoire**. Seuls 2 matchs (1925299 et 1996435) ont des données Body Pose ; aucune conclusion n'est attendue, l'objectif est de montrer ce que l'on pourrait tester avec davantage de matchs.

## Définitions
- Orientation du tronc : angle du vecteur d'épaules (25 images/s, rattaché au tracking 10 Hz par `T = pose_frame*2//5`), orienté par la direction de course.
- « Tourné vers son but » (`a_turned`) : défenseur de l'équipe qui défend, proche du ballon (< 10 m), tronc orienté vers son propre but.
- « Face au jeu » (`b_forward`) : attaquant proche du ballon, tronc orienté vers le but adverse.

## Hypothèses
- **H1** : plus de défenseurs proches du ballon tournés vers leur propre but ⇒ plus de tirs de l'adversaire dans les 15 s.
- **H2** : plus d'attaquants proches du ballon face au jeu ⇒ plus de tirs de l'adversaire dans les 15 s.

## Puissance
Avec 46 tirs adverses et 15 tirs de l'équipe sur 2 matchs, l'effet minimal détectable (MDE = 2,8 × erreur-type) est de 1,0 à 2,2 en rapport de cotes log-échelle : seuls de très gros effets seraient visibles. Un résultat nul ne réfute donc rien. Environ 10 matchs seraient nécessaires pour les tirs adverses (H1, H2).

## Règle de lecture
Résultat rapporté avec son intervalle et sa puissance, étiqueté « exploratoire », quel que soit le sens.


## Addendum du 20 octobre 2026 (post hoc, ne modifie pas les hypothèses)
- Les hypothèses H1 et H2, leur définition et leur règle de lecture sont inchangées ; le seuil de qualité enregistré reste **20 cm** (erreur p90 des épaules).
- `highline/pose.py` affiche désormais la part des joueurs proches avec une pose valide et une **analyse de sensibilité post hoc** (10, 20, 30 cm). Elle n'est pas une hypothèse de départ et ne sert pas à choisir un seuil.
- Trois mesures sont gardées séparées : la pose (observée, avec son erreur), l'orientation du buste (interprétée : perpendiculaire à la ligne des épaules, **pas un regard**) et l'identité (joueur de la pose = joueur du tracking par `player_id`). Un joueur sans pose valide est exclu, jamais imputé.
- La distance « à son but » part maintenant de la **ligne de but** (et non de la position médiane du gardien, décalée de 9 à 18 m), ce qui change l'ensemble des moments retenus, donc les effectifs de H1 et H2 ; les tests sont relancés tels quels sur la nouvelle définition.
