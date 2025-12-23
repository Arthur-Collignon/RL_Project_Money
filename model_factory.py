#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm

# Core SB3 algos (toujours présents en général)
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC, TD3

from stable_baselines3.common.noise import NormalActionNoise


# ============================================================
# Config (simple, modifiable)
# ============================================================

@dataclass
class SB3Config:
    # Common
    policy: str = "MlpPolicy"
    seed: int = 0
    device: str = "cpu"
    verbose: int = 1
    tensorboard_log: Optional[str] = None
    total_timesteps: int = 50_000

    # Common RL params
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # PPO / A2C
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.02

    # Off-policy (DQN/DDPG/TD3/SAC)
    buffer_size: int = 300_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005

    # DQN
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.02

    # Policy kwargs (optionnel)
    policy_kwargs: Optional[Dict[str, Any]] = None

    # Extra algorithm-specific overrides (optionnel)
    # ex: extra={"ppo": {"clip_range": 0.3}, "sac": {"ent_coef": "auto"}}
    extra: Optional[Dict[str, Dict[str, Any]]] = None


def _merge_algo_overrides(cfg: SB3Config, algo: str) -> Dict[str, Any]:
    """Fusionne cfg + cfg.extra[algo] sans casser le dataclass."""
    d = cfg.__dict__.copy()
    d.pop("extra", None)
    overrides = (cfg.extra or {}).get(algo, {})
    d.update(overrides)
    return d


# ============================================================
# Discovery / filtering
# ============================================================

def available_algorithms(include_contrib: bool = True) -> Dict[str, Any]:
    """
    Retourne {name: AlgoClass} pour les algos installés.
    - Core SB3: a2c, ppo, dqn, ddpg, sac, td3
    - Contrib (si installé): trpo, tqc, qrdqn, ars, recurrentppo, crossq, etc.
    """
    algos: Dict[str, Any] = {
        "a2c": A2C,
        "ppo": PPO,
        "dqn": DQN,
        "ddpg": DDPG,
        "sac": SAC,
        "td3": TD3,
    }

    if include_contrib:
        try:
            import sb3_contrib  # type: ignore

            # On ajoute seulement ceux qui existent dans la version installée.
            for name in ["TRPO", "TQC", "QRDQN", "ARS", "RecurrentPPO", "CrossQ"]:
                if hasattr(sb3_contrib, name):
                    algos[name.lower()] = getattr(sb3_contrib, name)
        except Exception:
            pass

    return algos


def supported_algorithms_for_env(env: gym.Env, include_contrib: bool = True) -> List[str]:
    """
    Filtre les algos selon l'action_space:
    - Discrete => DQN + PPO/A2C (+ TRPO/QRDQN/RecurrentPPO si présents)
    - Box => PPO/A2C + SAC/TD3/DDPG (+ TQC/CrossQ/ARS si présents)
    """
    algos = available_algorithms(include_contrib=include_contrib)
    a = env.action_space

    out: List[str] = []
    for name in algos.keys():
        if isinstance(a, spaces.Discrete):
            if name in {"dqn", "ppo", "a2c", "trpo", "qrdqn", "recurrentppo"}:
                out.append(name)
        elif isinstance(a, spaces.Box):
            if name in {"ppo", "a2c", "sac", "td3", "ddpg", "tqc", "crossq", "ars", "trpo", "recurrentppo"}:
                # trpo/recurrentppo supportent Box selon versions/implémentations
                out.append(name)
        else:
            # MultiDiscrete/MultiBinary etc: PPO/A2C supportent souvent; DQN non.
            if name in {"ppo", "a2c", "trpo", "recurrentppo"}:
                out.append(name)

    return sorted(set(out))


# ============================================================
# Factory: model_choice(algo, config, env) -> model
# ============================================================

def model_choice(algorithm: str, config: Union[SB3Config, Dict[str, Any]], env: gym.Env) -> BaseAlgorithm:
    """
    algorithm: "ppo"|"a2c"|"dqn"|"sac"|"td3"|"ddpg"| + contrib si installé
    config: SB3Config ou dict compatible
    env: env SB3 (VecEnv ou gym env)
    """
    if isinstance(config, dict):
        cfg = SB3Config(**config)
    else:
        cfg = config

    algo = algorithm.lower().strip()
    algo_map = available_algorithms(include_contrib=True)

    if algo not in algo_map:
        raise ValueError(f"Algo '{algorithm}' indisponible. Dispo: {sorted(algo_map.keys())}")

    # Applique overrides par algo si fournis
    d = _merge_algo_overrides(cfg, algo)

    # Policy kwargs (si fourni)
    policy_kwargs = d.get("policy_kwargs", None)

    common = dict(
        policy=d["policy"],
        env=env,
        learning_rate=d["learning_rate"],
        gamma=d["gamma"],
        seed=d["seed"],
        verbose=d["verbose"],
        device=d["device"],
        tensorboard_log=d["tensorboard_log"],
        policy_kwargs=policy_kwargs,
    )

    # Action noise auto pour DDPG/TD3 (continuous)
    action_noise = None
    if algo in {"ddpg", "td3"} and isinstance(env.action_space, spaces.Box):
        n_actions = int(np.prod(env.action_space.shape))
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    AlgoClass = algo_map[algo]

    # Cas core algos (on passe les params attendus)
    if algo == "ppo":
        return AlgoClass(
            **common,
            n_steps=d["n_steps"],
            batch_size=d["batch_size"],
            n_epochs=d["n_epochs"],
            gae_lambda=d["gae_lambda"],
            clip_range=d["clip_range"],
            ent_coef=d["ent_coef"],
            vf_coef=d["vf_coef"],
            max_grad_norm=d["max_grad_norm"],
            target_kl=d["target_kl"],
        )

    if algo == "a2c":
        return AlgoClass(
            **common,
            n_steps=d["n_steps"],
            gae_lambda=d["gae_lambda"],
            ent_coef=d["ent_coef"],
            vf_coef=d["vf_coef"],
            max_grad_norm=d["max_grad_norm"],
        )

    if algo == "dqn":
        return AlgoClass(
            **common,
            buffer_size=d["buffer_size"],
            learning_starts=d["learning_starts"],
            batch_size=d["batch_size"],
            train_freq=d["train_freq"],
            gradient_steps=d["gradient_steps"],
            exploration_fraction=d["exploration_fraction"],
            exploration_final_eps=d["exploration_final_eps"],
        )

    if algo == "sac":
        # ent_coef peut être "auto" (string) en SAC
        return AlgoClass(
            **common,
            buffer_size=d["buffer_size"],
            learning_starts=d["learning_starts"],
            batch_size=d["batch_size"],
            train_freq=d["train_freq"],
            gradient_steps=d["gradient_steps"],
            tau=d["tau"],
            ent_coef=d.get("ent_coef", "auto"),
        )

    if algo == "td3":
        return AlgoClass(
            **common,
            buffer_size=d["buffer_size"],
            learning_starts=d["learning_starts"],
            batch_size=d["batch_size"],
            train_freq=d["train_freq"],
            gradient_steps=d["gradient_steps"],
            tau=d["tau"],
            action_noise=action_noise,
        )

    if algo == "ddpg":
        return AlgoClass(
            **common,
            buffer_size=d["buffer_size"],
            learning_starts=d["learning_starts"],
            batch_size=d["batch_size"],
            train_freq=d["train_freq"],
            gradient_steps=d["gradient_steps"],
            tau=d["tau"],
            action_noise=action_noise,
        )

    # Cas contrib: on passe uniquement "common" + overrides (si tu veux du fin tuning)
    # (ça évite de casser si ta version n’a pas exactement les mêmes paramètres)
    extra_kwargs = (cfg.extra or {}).get(algo, {})
    extra_kwargs = {k: v for k, v in extra_kwargs.items() if k not in common}
    return AlgoClass(**common, **extra_kwargs)


# ============================================================
# Test all: entraîne et renvoie des résultats par algo
# ============================================================

def test_all_algorithms(
    make_env: Callable[[], gym.Env],
    config: Union[SB3Config, Dict[str, Any]],
    algorithms: Optional[List[str]] = None,
    include_contrib: bool = True,
    train_timesteps: Optional[int] = None,
    callbacks_factory: Optional[Callable[[str], Any]] = None,
    eval_fn: Optional[Callable[[BaseAlgorithm, gym.Env], Dict[str, float]]] = None,
    fail_fast: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    - make_env(): fonction qui construit un env (déjà VecEnv/VecNormalize si tu veux)
    - algorithms: liste d'algos à tester (None => auto selon action_space)
    - callbacks_factory(algo)-> callback ou liste de callbacks
    - eval_fn(model, env)-> dict métriques (optionnel)
    """
    env0 = make_env()
    try:
        supported = supported_algorithms_for_env(env0, include_contrib=include_contrib)
    finally:
        try:
            env0.close()
        except Exception:
            pass

    algo_list = algorithms or supported

    # total_timesteps
    if isinstance(config, dict):
        cfg_obj = SB3Config(**config)
    else:
        cfg_obj = config

    timesteps = int(train_timesteps if train_timesteps is not None else cfg_obj.total_timesteps)

    results: Dict[str, Dict[str, Any]] = {}

    for algo in algo_list:
        env = make_env()
        try:
            model = model_choice(algo, cfg_obj, env)

            cb = None
            if callbacks_factory is not None:
                c = callbacks_factory(algo)
                cb = c

            model.learn(total_timesteps=timesteps, callback=cb)

            metrics = {}
            if eval_fn is not None:
                metrics = eval_fn(model, env)

            results[algo] = {
                "status": "ok",
                "metrics": metrics,
            }

        except Exception as e:
            results[algo] = {
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }
            if fail_fast:
                raise
        finally:
            try:
                env.close()
            except Exception:
                pass

    return results
