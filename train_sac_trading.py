#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import glob
import random
import shutil
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import gymnasium as gym

import wandb
from wandb.integration.sb3 import WandbCallback

# IMPORTANT: nécessaire pour enregistrer l'env "MultiDatasetTradingEnv"
import gym_trading_env  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


# ============================================================
# Config
# ============================================================

@dataclass
class ExperimentConfig:
    # Data
    data_dir: str = "data"
    seed: int = 0

    # Split fixé => runs comparables
    split_seed: int = 0
    test_ratio: float = 0.35
    min_rows_after_preprocess: int = 300
    min_test_files: int = 2

    # Env
    portfolio_initial_value: float = 1000.0
    trading_fees: float = 0.1 / 100
    borrow_interest_rate: float = 0.02 / 100 / 24
    position_range: Tuple[float, float] = (0.0, 1.0)  # long-only
    # si l'énoncé le permet, teste (-1, 1) pour battre buy&hold en bull market :
    # position_range: Tuple[float, float] = (-1.0, 1.0)

    # Reward shaping (inspiré du projet.ipynb)
    # - "default": reward interne env
    # - "log_return": log(V_t/V_{t-1})
    # - "alpha_log_return": log(V_t/V_{t-1}) - log(P_t/P_{t-1})
    reward_mode: str = "alpha_log_return"

    # Actions discrètes (inspiré du projet.ipynb)
    use_discrete_actions: bool = False
    discrete_positions: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0)

    # Anti-overtrading
    turnover_lambda: float = 1e-3

    # VecNormalize
    n_envs: int = 4
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0

    # Train
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    model_dir: str = "models"

    # PPO
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.02

    # Policy
    net_arch_pi: Tuple[int, int] = (256, 256)
    net_arch_vf: Tuple[int, int] = (256, 256)

    # Eval seeds FIXES (comparabilité)
    eval_seed_base: int = 12345
    deterministic_eval: bool = True

    # W&B
    wandb_project: str = "rl-trading-env"
    wandb_entity: Optional[str] = None
    wandb_sync_tensorboard: bool = True


# ============================================================
# Wrapper anti-overtrading
# ============================================================

class TurnoverPenaltyWrapper(gym.Wrapper):
    """
    reward' = reward - lam * mean(|a_t - a_{t-1}|)
    Important: appliquer AVANT DiscreteActionsWrapper, pour pénaliser la position réelle.
    """
    def __init__(self, env: gym.Env, lam: float = 1e-3):
        super().__init__(env)
        self.lam = float(lam)
        self.prev_action: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # action continue attendue (Box)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        a = np.array(action, dtype=np.float32)
        if self.prev_action is None:
            self.prev_action = np.zeros_like(a)

        penalty = self.lam * float(np.abs(a - self.prev_action).mean())
        self.prev_action = a

        info = dict(info)
        info["turnover_penalty"] = penalty
        return obs, reward - penalty, terminated, truncated, info


# ============================================================
# History accessor (robuste)
# ============================================================

def hget(history: Any, key: str, idx: int) -> Optional[float]:
    """
    Essayes plusieurs formats possibles de 'history' (gym_trading_env varie selon versions).
    """
    # format "history['portfolio_valuation', -1]" vu dans projet.ipynb
    try:
        v = history[key, idx]
        return float(v)
    except Exception:
        pass

    # pandas DataFrame/Series
    try:
        if hasattr(history, "columns") and key in history.columns:
            return float(history[key].iloc[idx])
    except Exception:
        pass

    # dict of lists / dict of arrays
    try:
        if isinstance(history, dict) and key in history:
            v = history[key]
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                return float(v[idx])
    except Exception:
        pass

    # list of dicts
    try:
        if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
            return float(history[idx].get(key))
    except Exception:
        pass

    return None


def find_price_key(history: Any) -> Optional[str]:
    # clés fréquentes possibles
    candidates = ["close", "price", "data_close", "asset_close", "market_price"]
    for k in candidates:
        v0 = hget(history, k, 0)
        v1 = hget(history, k, -1)
        if v0 is not None and v1 is not None:
            return k
    return None


# ============================================================
# Reward shaping (projet.ipynb)
# ============================================================

def make_reward_function(mode: str) -> Optional[Callable[[Any], float]]:
    mode = (mode or "default").lower().strip()
    if mode == "default":
        return None

    def _reward(history: Any) -> float:
        pv_t = hget(history, "portfolio_valuation", -1)
        pv_tm1 = hget(history, "portfolio_valuation", -2)

        if pv_t is None:
            # fallback très conservateur
            return 0.0

        if mode == "portfolio_value":
            return float(pv_t)

        # log return portefeuille
        if pv_tm1 is None or pv_tm1 <= 0 or pv_t <= 0:
            r_port = 0.0
        else:
            r_port = float(np.log(pv_t / pv_tm1))

        if mode == "log_return":
            return r_port

        if mode == "alpha_log_return":
            pk = find_price_key(history)
            if pk is None:
                return r_port
            p_t = hget(history, pk, -1)
            p_tm1 = hget(history, pk, -2)
            if p_t is None or p_tm1 is None or p_t <= 0 or p_tm1 <= 0:
                return r_port
            r_mkt = float(np.log(p_t / p_tm1))
            return r_port - r_mkt

        # fallback
        return r_port

    return _reward


# ============================================================
# Metrics (projet.ipynb)
# ============================================================

def add_default_metrics(env: gym.Env) -> None:
    """
    Ajoute des métriques via env.add_metric si dispo.
    """
    if not hasattr(env, "add_metric"):
        return

    def metric_portfolio_valuation(history):
        v = hget(history, "portfolio_valuation", -1)
        return float("nan") if v is None else round(float(v), 2)

    def metric_portfolio_return(history):
        v0 = hget(history, "portfolio_valuation", 0)
        v1 = hget(history, "portfolio_valuation", -1)
        if v0 is None or v1 is None or v0 == 0:
            return float("nan")
        return round(float(v1 / v0 - 1.0), 4)

    def metric_market_return(history):
        pk = find_price_key(history)
        if pk is None:
            return float("nan")
        p0 = hget(history, pk, 0)
        p1 = hget(history, pk, -1)
        if p0 is None or p1 is None or p0 == 0:
            return float("nan")
        return round(float(p1 / p0 - 1.0), 4)

    env.add_metric("Portfolio Valuation", metric_portfolio_valuation)
    env.add_metric("Portfolio Return", metric_portfolio_return)
    env.add_metric("Market Return", metric_market_return)


# ============================================================
# Features / preprocess (RSI-MACD + robust)
# ============================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Ajoute RSI/MACD (comme RSI-MACD.ipynb) mais version causale/robuste
    - Remplit NaN des features (évite de tuer le dataset)
    """
    df = df.sort_index().drop_duplicates()

    if "close" not in df.columns:
        raise ValueError(f"Dataset sans colonne 'close'. Colonnes: {list(df.columns)[:20]}")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["close"] = df["close"].ffill().bfill()
    df = df.dropna(subset=["close"])

    close = df["close"].astype(float)

    # Ajout d'une feature prix (utile pour debug)
    df["feature_close"] = close

    # returns/vol
    df["feature_log_return"] = np.log(close).diff()
    df["feature_volatility_24"] = df["feature_log_return"].rolling(24, min_periods=24).std()

    # RSI (14 + 24 comme dans l'idée 1h -> 1 jour)
    df["feature_rsi_14"] = _rsi(close, 14)
    df["feature_rsi_24"] = _rsi(close, 24)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["feature_macd"] = macd
    df["feature_macd_signal"] = signal
    df["feature_macd_hist"] = macd - signal

    # Volume
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").replace(0, np.nan)
        df["feature_log_volume"] = np.log(vol)
    else:
        df["feature_log_volume"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    df[feat_cols] = df[feat_cols].fillna(0.0)
    return df


def preprocess_and_cache_pkls(src_files: List[str], out_dir: str, min_rows: int) -> List[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    kept: List[str] = []
    skipped = 0

    for f in src_files:
        try:
            df = pd.read_pickle(f)
            df2 = preprocess_df(df)
            if len(df2) < min_rows:
                skipped += 1
                continue
            out_path = str(Path(out_dir) / Path(f).name)
            df2.to_pickle(out_path)
            kept.append(out_path)
        except Exception:
            skipped += 1

    print(f"[cache] kept={len(kept)} skipped={skipped} -> {out_dir}")
    if not kept:
        raise RuntimeError("Aucun dataset valide après preprocess.")
    return kept


def split_after_preprocess(cfg: ExperimentConfig) -> Tuple[str, str, List[str], List[str]]:
    raw_files = sorted(glob.glob(os.path.join(cfg.data_dir, "*.pkl")))
    if not raw_files:
        raise FileNotFoundError(f"Aucun .pkl trouvé dans {cfg.data_dir}/*.pkl")

    all_cache_dir = os.path.join(cfg.data_dir, "_preprocessed_all")
    cached_files = preprocess_and_cache_pkls(raw_files, all_cache_dir, cfg.min_rows_after_preprocess)

    random.seed(cfg.split_seed)
    random.shuffle(cached_files)

    n_test = max(cfg.min_test_files, int(cfg.test_ratio * len(cached_files)))
    n_test = min(n_test, len(cached_files) - 1)

    test_files = cached_files[:n_test]
    train_files = cached_files[n_test:]

    train_dir = os.path.join(cfg.data_dir, "_train_preprocessed")
    test_dir = os.path.join(cfg.data_dir, "_test_preprocessed")
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    # clean
    for d in [train_dir, test_dir]:
        for old in glob.glob(os.path.join(d, "*.pkl")):
            try:
                os.remove(old)
            except OSError:
                pass

    for f in train_files:
        shutil.copy2(f, train_dir)
    for f in test_files:
        shutil.copy2(f, test_dir)

    train_glob = str(Path(train_dir) / "*.pkl")
    test_glob = str(Path(test_dir) / "*.pkl")
    print(f"[split] train_files={len(train_files)} test_files={len(test_files)}")
    return train_glob, test_glob, train_files, test_files


# ============================================================
# Env
# ============================================================

def make_single_env(dataset_pattern: str, cfg: ExperimentConfig) -> gym.Env:
    reward_fn = make_reward_function(cfg.reward_mode)

    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir=dataset_pattern,
        preprocess=lambda df: df,  # déjà préprocessé/caché
        portfolio_initial_value=cfg.portfolio_initial_value,
        trading_fees=cfg.trading_fees,
        borrow_interest_rate=cfg.borrow_interest_rate,
        position_range=cfg.position_range,
        **({"reward_function": reward_fn} if reward_fn is not None else {}),
    )

    # Metrics style projet.ipynb
    add_default_metrics(env)

    # Turnover penalty (avant DiscreteActionsWrapper)
    env = TurnoverPenaltyWrapper(env, lam=cfg.turnover_lambda)

    # Actions discrètes optionnelles (projet.ipynb)
    if cfg.use_discrete_actions:
        from gym_trading_env.wrapper import DiscreteActionsWrapper
        env = DiscreteActionsWrapper(env, positions=list(cfg.discrete_positions))

    return env


def make_env_fn(dataset_pattern: str, cfg: ExperimentConfig, rank: int):
    def _init():
        env = make_single_env(dataset_pattern, cfg)
        env.reset(seed=cfg.seed + rank)
        return env
    return _init


# ============================================================
# Eval helpers
# ============================================================

def evaluate_on_vec_env(
    vec_env: VecNormalize,
    act_fn: Callable[[np.ndarray], np.ndarray],
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    finals: List[float] = []

    for ep in range(n_episodes):
        vec_env.seed(seed + ep)
        obs = vec_env.reset()

        done = False
        last_val = None

        while not done:
            action = act_fn(obs)
            obs, rewards, dones, infos = vec_env.step(action)
            done = bool(dones[0])

            if infos and isinstance(infos[0], dict) and "portfolio_valuation" in infos[0]:
                last_val = float(infos[0]["portfolio_valuation"])

        # fallback sur historique
        if last_val is None:
            try:
                hist = vec_env.get_attr("historical_info")[0]
                v = hget(hist, "portfolio_valuation", -1)
                if v is not None:
                    last_val = float(v)
            except Exception:
                pass

        if last_val is not None:
            finals.append(last_val)

    if not finals:
        return {"mean_final": float("nan"), "std_final": float("nan"), "n": 0.0}

    return {
        "mean_final": float(np.mean(finals)),
        "std_final": float(np.std(finals)),
        "n": float(len(finals)),
    }


def eval_by_file_table(cfg: ExperimentConfig, model: PPO, train_obs_rms, test_files: List[str]):
    table = wandb.Table(columns=["file", "policy", "mean_final", "std_final", "n"])

    ppo_wins = 0
    n_files = 0
    ppo_means, bh_means, rnd_means = [], [], []

    for fpath in test_files:
        one_eval = DummyVecEnv([make_env_fn(fpath, cfg, 10_000)])
        one_eval = VecMonitor(one_eval)
        one_eval = VecNormalize(
            one_eval, training=False,
            norm_obs=cfg.norm_obs, norm_reward=False, clip_obs=cfg.clip_obs
        )
        one_eval.obs_rms = train_obs_rms

        def random_act(_obs):
            # Discrete/Box géré par action_space.sample()
            a = one_eval.action_space.sample()
            # DummyVecEnv attend shape (n_env, ...)
            return np.array([a])

        def buyhold_act(_obs):
            # si discret: pick l'index de la position max
            if cfg.use_discrete_actions:
                # position max = last index
                return np.array([len(cfg.discrete_positions) - 1])
            return np.array([[1.0]], dtype=np.float32)

        def ppo_act(obs):
            a, _ = model.predict(obs, deterministic=cfg.deterministic_eval)
            return a

        base = cfg.eval_seed_base
        rr = evaluate_on_vec_env(one_eval, random_act, cfg.n_eval_episodes, base + 1_000)
        bh = evaluate_on_vec_env(one_eval, buyhold_act, cfg.n_eval_episodes, base + 2_000)
        pp = evaluate_on_vec_env(one_eval, ppo_act, cfg.n_eval_episodes, base + 3_000)

        fname = os.path.basename(fpath)
        table.add_data(fname, "random", rr["mean_final"], rr["std_final"], int(rr["n"]))
        table.add_data(fname, "buyhold", bh["mean_final"], bh["std_final"], int(bh["n"]))
        table.add_data(fname, "ppo", pp["mean_final"], pp["std_final"], int(pp["n"]))

        if np.isfinite(rr["mean_final"]): rnd_means.append(rr["mean_final"])
        if np.isfinite(bh["mean_final"]): bh_means.append(bh["mean_final"])
        if np.isfinite(pp["mean_final"]): ppo_means.append(pp["mean_final"])

        if np.isfinite(pp["mean_final"]) and np.isfinite(bh["mean_final"]):
            n_files += 1
            if pp["mean_final"] > bh["mean_final"]:
                ppo_wins += 1

        one_eval.close()

    def smean(x): return float(np.mean(x)) if x else float("nan")

    summary = {
        "test/random_mean_final": smean(rnd_means),
        "test/buyhold_mean_final": smean(bh_means),
        "test/ppo_mean_final": smean(ppo_means),
    }
    summary["test/ppo_minus_buyhold"] = summary["test/ppo_mean_final"] - summary["test/buyhold_mean_final"]
    summary["test/win_rate_vs_buyhold"] = (ppo_wins / n_files) if n_files > 0 else float("nan")
    return table, summary


# ============================================================
# Main training
# ============================================================

def run_experiment(cfg: ExperimentConfig, group: Optional[str] = None) -> Dict[str, float]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    print("[config]")
    for k, v in asdict(cfg).items():
        print(f"  - {k}: {v}")

    train_glob, test_glob, train_files, test_files = split_after_preprocess(cfg)

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=asdict(cfg),
        save_code=True,
        sync_tensorboard=cfg.wandb_sync_tensorboard,
        group=group,
    )

    # run name lisible
    wandb.run.name = (
        f"seed={cfg.seed}_split={cfg.split_seed}_reward={cfg.reward_mode}"
        f"_disc={int(cfg.use_discrete_actions)}_lam={cfg.turnover_lambda:g}_lr={cfg.learning_rate:g}"
    )

    # log fichiers test (utile pour debug)
    wandb.log({
        "data/n_train_files": len(train_files),
        "data/n_test_files": len(test_files),
        "data/test_files_table": wandb.Table(columns=["file"], data=[[os.path.basename(f)] for f in test_files]),
    })

    # Train env
    train_env = DummyVecEnv([make_env_fn(train_glob, cfg, i) for i in range(cfg.n_envs)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        training=True,
        norm_obs=cfg.norm_obs,
        norm_reward=cfg.norm_reward,
        clip_obs=cfg.clip_obs,
    )

    # Eval env (pour courbes eval/* via EvalCallback)
    eval_env = DummyVecEnv([make_env_fn(test_glob, cfg, 10_000)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=cfg.norm_obs,
        norm_reward=False,
        clip_obs=cfg.clip_obs,
    )
    eval_env.obs_rms = train_env.obs_rms

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg.model_dir,
        log_path=cfg.model_dir,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=cfg.deterministic_eval,
    )

    callback = CallbackList([
        eval_cb,
        WandbCallback(
            model_save_path=os.path.join(cfg.model_dir, "wandb_models", run.id),
            model_save_freq=0,
            verbose=0,
        ),
    ])

    tb_dir = os.path.join(cfg.model_dir, "tb", run.id) if cfg.wandb_sync_tensorboard else None
    policy_kwargs = dict(net_arch=dict(pi=list(cfg.net_arch_pi), vf=list(cfg.net_arch_vf)))

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        target_kl=cfg.target_kl,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg.seed,
        device="cpu",
        tensorboard_log=tb_dir,
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

    # Post-train eval (log en W&B)
    by_file_table, summary = eval_by_file_table(cfg, model, train_env.obs_rms, test_files)
    wandb.log({"test/by_file_table": by_file_table, **summary})
    wandb.run.summary["test/by_file_table"] = by_file_table
    for k, v in summary.items():
        wandb.run.summary[k] = v

    # save model + normalizer
    model_path = os.path.join(cfg.model_dir, f"ppo_{run.id}")
    norm_path = os.path.join(cfg.model_dir, f"vecnormalize_{run.id}.pkl")
    model.save(model_path)
    train_env.save(norm_path)
    wandb.run.summary["model_path"] = model_path + ".zip"
    wandb.run.summary["vecnormalize_path"] = norm_path

    wandb.finish()
    eval_env.close()
    train_env.close()
    return summary


# ============================================================
# Example: run grid quick
# ============================================================

def run_small_grid():
    base = ExperimentConfig(
        data_dir="data",
        seed=0,
        split_seed=0,
        reward_mode="alpha_log_return",
        use_discrete_actions=False,   # teste True si PPO instable
        total_timesteps=20_000,
        eval_freq=10_000,
        n_eval_episodes=5,
        n_envs=4,
        turnover_lambda=1e-3,
        learning_rate=3e-4,
        ent_coef=0.01,
        vf_coef=1.0,
        target_kl=0.02,
        wandb_sync_tensorboard=True,
    )

    seeds = [0, 10, 20]
    lams = [1e-4, 1e-3, 5e-3]

    for s in seeds:
        for lam in lams:
            cfg = deepcopy(base)
            cfg.seed = s
            cfg.turnover_lambda = lam
            run_experiment(cfg, group="grid_reward_alpha")


if __name__ == "__main__":
    # Run unique:
    # cfg = ExperimentConfig()
    # run_experiment(cfg)

    # Grid:
    run_small_grid()
