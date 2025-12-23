# RL Trading Project : Agent Autonome sur Indices Financiers

Ce projet explore l'application du Deep Reinforcement Learning (DRL) au trading algorithmique. L'objectif était de concevoir un agent capable de gérer le risque et de générer de l'Alpha sur des données financières réelles.

## Liens du Projet

* **Code Source (GitHub) :** [https://github.com/Arthur-Collignon/RL_Project_Money](https://github.com/Arthur-Collignon/RL_Project_Money)
* **Suivi des Expériences (WandB) :** [Dashboard Weights & Biases](https://wandb.ai/thomas-derville-cpe-lyon/RL-Trading-Project?nw=nwuserarthurcollignon)
  *(Contient l'historique des courbes de reward, pertes et métriques de tous les modèles testés)*

## Organisation du Code

La démarche scientifique et le résultat final sont séparés en deux fichiers principaux :

* **`project.ipynb` : Historique et Réflexion**
    * Ce notebook contient **le détail de la réflexion** et l'intégralité du processus de R&D.
    * Vous y trouverez les différentes phases du projet : les échecs initiaux, les tests sur différents algorithmes (TD3, SAC, DQN, A2C), l'implémentation des modèles récurrents (LSTM) et les ajustements de la fonction de récompense.

* **`rendu_final.ipynb` : Modèle Final**
    * Ce notebook contient le **rendu final** épuré.
    * Il présente l'architecture retenue (**PPO** avec *Windowed Observation* et Normalisation Z-Score) qui offre la meilleure stabilité et gestion du risque.
    * Il inclut le code d'entraînement, d'évaluation et de visualisation.