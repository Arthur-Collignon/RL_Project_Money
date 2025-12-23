# RL Trading Project : Agent Autonome sur Indices Financiers

Ce projet explore l'application du Deep Reinforcement Learning (DRL) au trading algorithmique. L'objectif était de concevoir un agent capable de gérer le risque et de générer de l'Alpha sur des données financières réelles.

## Suivi des Expérimentations (WandB)

L'ensemble des logs d'entraînement, courbes de reward, pertes et métriques de performance de toutes les itérations sont accessibles via le dashboard Weights & Biases :

**[Voir le Dashboard WandB du Projet](https://wandb.ai/thomas-derville-cpe-lyon/RL-Trading-Project?nw=nwuserarthurcollignon)**

## Organisation du Code

La démarche scientifique et le résultat final sont séparés en deux fichiers distincts :

* **`project.ipynb` : Historique et Réflexion**
    * Ce notebook contient **le détail de la réflexion** et l'intégralité du processus de R&D.
    * Vous y trouverez les différentes phases du projet : les échecs initiaux, les tests sur différents algorithmes (TD3, SAC, DQN, A2C), l'implémentation des modèles récurrents (LSTM) et les ajustements de la fonction de récompense.

* **`rendu_final.ipynb` : Modèle Final**
    * Ce notebook contient le **rendu final** épuré.
    * Il présente l'architecture retenue (**PPO** avec *Windowed Observation* et Normalisation Z-Score) qui offre la meilleure stabilité et gestion du risque.
    * Il inclut le code d'entraînement, d'évaluation et de visualisation.