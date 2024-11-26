# LWFA-simulation
Ce projet implémente une simulation 2D haute performance de l'accélération d'électrons dans un plasma à l'aide de l'effet Laser Wakefield (LWFA).
Les fonctionnalités incluent :

  - Un champ électrique laser avec profil gaussien dynamique.
  - La résolution de l'équation de Poisson pour calculer les champs électriques du plasma.
  - Une gestion optimisée des particules simulant les électrons, incluant leur déposition de charge, interpolation de champs, et mise à jour des vitesses et positions.
     Une visualisation en temps réel des positions des électrons, de l'intensité du champ laser, et de la distribution d'énergie des électrons.

Technologies :

    Python (Numba, CUDA, CuPy)
    Matplotlib pour la visualisation
    Modèles physiques avancés et parallélisme GPU.

Usage :
Conçu pour la recherche scientifique, l'enseignement, et les démonstrations visuelles des phénomènes physiques liés à l'accélération d'électrons dans les plasmas.
