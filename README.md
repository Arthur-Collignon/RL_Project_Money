# RL Trading Project : Agent Autonome sur Indices Financiers

Ce projet explore l'application du Deep Reinforcement Learning (DRL) au trading algorithmique. L'objectif était de concevoir un agent capable de gérer le risque et de générer de l'Alpha sur des données financières réelles.

## Liens du Projet

* **Code Source (GitHub) :** [https://github.com/Arthur-Collignon/RL_Project_Money](https://github.com/Arthur-Collignon/RL_Project_Money)
* **Suivi des Expériences (WandB) :** [Dashboard Weights & Biases](https://wandb.ai/thomas-derville-cpe-lyon/RL-Trading-Project/reports/D-veloppement-d-un-Agent-de-Trading-par-Apprentissage-par-Renforcement--VmlldzoxNTQ1MzAwNg?accessToken=ox4b8kylnp6pco5ff0q5h0l3mf2qau65ric7v48g8xfx2eetmbuonreiujpny6q4)
  *(Contient l'historique des courbes de reward, pertes et métriques de tous les modèles testés)*

## Organisation du Code

La démarche scientifique et le résultat final sont séparés en deux fichiers principaux :

* **`project.ipynb` : Historique et Réflexion**
    * Ce notebook contient **le détail de la réflexion** et l'intégralité du processus de R&D.
    * Vous y trouverez les différentes phases du projet : les échecs initiaux, les tests sur différents algorithmes (TD3, SAC, DQN, A2C), l'implémentation des modèles récurrents (LSTM) et les ajustements de la fonction de récompense.

    * **`CleanProject.ipynb` : Modèle final sous forme de Script Python**
    * Ce script contient le rendu final dans un script Python exécutable mais avec des hyperparamètres prédéfinis.

* **`rendu_final.ipynb` : Modèle Final**
    * Ce notebook contient le **rendu final** épuré.
    * Il présente l'architecture retenue (**PPO** avec *Windowed Observation* et Normalisation Z-Score) qui offre la meilleure stabilité et gestion du risque.
    * Il inclut le code d'entraînement, d'évaluation et de visualisation.
    * La fonction de reward a été modifiée par rapport à CleanProject pour mieux gérer le risque.