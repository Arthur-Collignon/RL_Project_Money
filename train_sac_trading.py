#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import glob
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
import wandb

# IMPORTANT: registre l'env "MultiDatasetTradingEnv"
import gym_trading_env  # noqa: F401

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


# ============================================================
# Import FACTORY (adapte automatiquement au nom du fichier)
# ============================================================

from model_factory import available_algorithms, model_choice, SB3Config  # type: ignore

# ============================================================
# Config "projet" (data/env/eval)
# ============================================================

@dataclass
class ProjectConfig:
    data_dir: str = "data"
    seed: int = 0
    split_seed: int = 0
    test_ratio: float = 0.35
    min_test_files: int = 2
    min_rows_after_preprocess: int = 300

    # Env params
    portfolio_initial_value: float = 1000.0
    trading_fees: float = 0.1 / 100
    borrow_interest_rate: float = 0.02 / 100 / 24
    position_range: Tuple[float, float] = (0.0, 1.0)

    # VecNormalize
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0

    # Vector env count (DummyVecEnv pour Windows)
    n_envs: int = 1

    # Eval
    n_eval_episodes: int = 5
    eval_seed_base: int = 12345
    eval_freq: int = 10_000  # pour EvalCallback (courbes eval/)

    # W&B
    wandb_project: str = "rl-trading-env"
    wandb_entity: Optional[str] = None
    wandb_group: str = "bench_all_algos"
    wandb_sync_tensorboard: bool = True

    # Output
    out_dir: str = "models"


# ============================================================
# Preprocess (simple + robuste)
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
    df = df.sort_index().drop_duplicates()

    if "close" not in df.columns:
        raise ValueError("Dataset sans colonne 'close'")

    df["close"] = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    df = df.dropna(subset=["close"])

    close = df["close"].astype(float)

    df["feature_log_return"] = np.log(close).diff()
    df["feature_volatility_24"] = df["feature_log_return"].rolling(24, min_periods=24).std()
    df["feature_rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["feature_macd"] = macd
    df["feature_macd_signal"] = signal
    df["feature_macd_hist"] = macd - signal

    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").replace(0, np.nan)
        df["feature_log_volume"] = np.log(vol)
    else:
        df["feature_log_volume"] = 0.0

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


def split_after_preprocess(cfg: ProjectConfig) -> Tuple[str, str, List[str], List[str]]:
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
# Env builders
# ============================================================

def make_base_env(dataset_pattern: str, pcfg: ProjectConfig) -> gym.Env:
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir=dataset_pattern,
        preprocess=lambda df: df,  # déjà préprocessé/caché
        portfolio_initial_value=pcfg.portfolio_initial_value,
        trading_fees=pcfg.trading_fees,
        borrow_interest_rate=pcfg.borrow_interest_rate,
        position_range=pcfg.position_range,
    )
    return env


def make_vec_env(
    dataset_pattern: str,
    pcfg: ProjectConfig,
    *,
    training: bool,
    obs_rms=None,
    discrete_positions: Optional[List[float]] = None,
) -> VecNormalize:
    """
    - DummyVecEnv (stable sur Windows)
    - VecNormalize obs (et reward uniquement en training)
    - optionnel: DiscreteActionsWrapper pour DQN/QRDQN
    """
    def make_one():
        e = make_base_env(dataset_pattern, pcfg)
        if discrete_positions is not None:
            from gym_trading_env.wrapper import DiscreteActionsWrapper
            e = DiscreteActionsWrapper(e, positions=discrete_positions)
        return e

    env_fns = [lambda: make_one() for _ in range(pcfg.n_envs)]
    venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)

    venv = VecNormalize(
        venv,
        training=training,
        norm_obs=pcfg.norm_obs,
        norm_reward=(pcfg.norm_reward if training else False),
        clip_obs=pcfg.clip_obs,
    )

    if obs_rms is not None:
        venv.obs_rms = obs_rms

    # seed stable
    try:
        venv.seed(pcfg.seed)
    except Exception:
        pass

    return venv


# ============================================================
# Evaluation (portfolio final) sur VecEnv
# ============================================================

def eval_mean_final_portfolio(vec_env: VecNormalize, act_fn, n_episodes: int, seed: int) -> float:
    finals: List[float] = []

    for ep in range(n_episodes):
        try:
            vec_env.seed(seed + ep)
        except Exception:
            pass

        obs = vec_env.reset()
        done = False
        last_val = None

        while not done:
            action = act_fn(obs)
            obs, rewards, dones, infos = vec_env.step(action)
            done = bool(dones[0])

            # tentative 1: info direct
            if infos and isinstance(infos[0], dict) and "portfolio_valuation" in infos[0]:
                try:
                    last_val = float(infos[0]["portfolio_valuation"])
                except Exception:
                    pass

        # fallback 2: metrics de l'env (projet.ipynb)
        if last_val is None:
            try:
                get_metrics = vec_env.get_attr("get_metrics")[0]
                metrics = get_metrics()
                # Selon env: "Portfolio Valuation" ou autre
                if isinstance(metrics, dict):
                    for k in ["Portfolio Valuation", "portfolio_valuation"]:
                        if k in metrics:
                            last_val = float(metrics[k])
                            break
            except Exception:
                pass

        # fallback 3: historical_info
        if last_val is None:
            try:
                hist = vec_env.get_attr("historical_info")[0]
                # hist peut être list[dict] ou DataFrame
                if isinstance(hist, list) and hist and isinstance(hist[-1], dict) and "portfolio_valuation" in hist[-1]:
                    last_val = float(hist[-1]["portfolio_valuation"])
                elif hasattr(hist, "columns") and "portfolio_valuation" in hist.columns:
                    last_val = float(hist["portfolio_valuation"].iloc[-1])
            except Exception:
                pass

        if last_val is not None and np.isfinite(last_val):
            finals.append(last_val)

    return float(np.mean(finals)) if finals else float("nan")


# ============================================================
# Hyperparams par algo (précis)
# ============================================================

def algo_hyperparams() -> Dict[str, Dict[str, Any]]:
    """
    Tu peux modifier ici et relancer.
    """
    return {
        # On-policy (actions Box ou Discrete)
        "ppo": dict(total_timesteps=30_000, learning_rate=3e-4, n_steps=2048, batch_size=256,
                    n_epochs=10, ent_coef=0.01, vf_coef=2.0, gae_lambda=0.95, clip_range=0.2, target_kl=0.02),
        "a2c": dict(total_timesteps=30_000, learning_rate=7e-4, n_steps=128, ent_coef=0.0, vf_coef=1.0, gae_lambda=0.95),

        # Off-policy continuous
        "sac": dict(total_timesteps=40_000, learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                    learning_starts=10_000, tau=0.005, train_freq=1, gradient_steps=1,
                    extra={"sac": {"ent_coef": "auto"}}),
        "td3": dict(total_timesteps=40_000, learning_rate=1e-3, buffer_size=300_000, batch_size=256,
                    learning_starts=10_000, tau=0.005, train_freq=1, gradient_steps=1),
        "ddpg": dict(total_timesteps=40_000, learning_rate=1e-3, buffer_size=300_000, batch_size=256,
                     learning_starts=10_000, tau=0.005, train_freq=1, gradient_steps=1),

        # Discrete only
        "dqn": dict(total_timesteps=40_000, learning_rate=1e-4, buffer_size=200_000, batch_size=64,
                    learning_starts=5_000, train_freq=4, gradient_steps=1,
                    exploration_fraction=0.2, exploration_final_eps=0.02),

        # Contrib (si présent)
        "trpo": dict(total_timesteps=30_000, learning_rate=3e-4),
        "recurrentppo": dict(total_timesteps=30_000, learning_rate=3e-4),
        "qrdqn": dict(total_timesteps=40_000, learning_rate=1e-4, buffer_size=200_000, batch_size=64,
                      learning_starts=5_000, train_freq=4, gradient_steps=1),
        "tqc": dict(total_timesteps=40_000, learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                    learning_starts=10_000, tau=0.005, train_freq=1, gradient_steps=1),
        "crossq": dict(total_timesteps=40_000, learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                       learning_starts=10_000, tau=0.005, train_freq=1, gradient_steps=1),
        "ars": dict(total_timesteps=40_000, learning_rate=3e-4),
    }


# ============================================================
# Main benchmark + W&B
# ============================================================

def main():
    pcfg = ProjectConfig(
        data_dir="data",
        seed=0,
        split_seed=0,
        test_ratio=0.35,
        min_test_files=2,
        min_rows_after_preprocess=300,
        position_range=(0.0, 1.0),
        n_envs=1,  # monte à 4 si tu veux, mais garde DummyVecEnv
        eval_freq=10_000,
        n_eval_episodes=5,
        eval_seed_base=12345,
        wandb_project="rl-trading-env",
        wandb_entity=None,  # "lafuethibaut-cpe-lyon" si tu veux forcer
        wandb_group="bench_all_algos",
        wandb_sync_tensorboard=True,
        out_dir="models",
    )

    print("[project config]")
    for k, v in asdict(pcfg).items():
        print(f"  - {k}: {v}")

    train_glob, test_glob, train_files, test_files = split_after_preprocess(pcfg)

    # algos installés via factory
    algos_map = available_algorithms(include_contrib=True)
    installed_algos = sorted(algos_map.keys())
    print(f"[algos installed] {installed_algos}")

    hp_map = algo_hyperparams()

    # positions discrètes pour DQN/QRDQN
    discrete_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    results: List[Dict[str, Any]] = []

    for algo in installed_algos:
        if algo not in hp_map:
            print(f"[skip] {algo}: pas d'hyperparams définis")
            continue

        hp = hp_map[algo]
        needs_discrete = ("dqn" in algo)  # dqn + qrdqn

        run = None
        train_env = None
        eval_env = None

        try:
            # W&B run
            run = wandb.init(
                project=pcfg.wandb_project,
                entity=pcfg.wandb_entity,
                group=pcfg.wandb_group,
                name=f"{algo}_seed{pcfg.seed}",
                config={
                    "project_cfg": asdict(pcfg),
                    "algo": algo,
                    "algo_hp": hp,
                    "train_files": [os.path.basename(f) for f in train_files],
                    "test_files": [os.path.basename(f) for f in test_files],
                },
                save_code=True,
                sync_tensorboard=pcfg.wandb_sync_tensorboard,
            )

            # TensorBoard dir (pour courbes rollout/train/eval)
            tb_dir = os.path.join(pcfg.out_dir, "tb", run.id)
            Path(tb_dir).mkdir(parents=True, exist_ok=True)

            print(f"\n===== {algo.upper()} =====")
            print(f"[wandb] {run.url}")

            # envs
            train_env = make_vec_env(
                train_glob, pcfg, training=True,
                discrete_positions=discrete_positions if needs_discrete else None,
            )
            eval_env = make_vec_env(
                test_glob, pcfg, training=False,
                obs_rms=train_env.obs_rms,
                discrete_positions=discrete_positions if needs_discrete else None,
            )

            # callbacks (eval -> log eval/* dans TB, donc sync vers W&B)
            callbacks = []
            if pcfg.eval_freq and pcfg.eval_freq > 0:
                eval_cb = EvalCallback(
                    eval_env,
                    eval_freq=pcfg.eval_freq,
                    n_eval_episodes=pcfg.n_eval_episodes,
                    deterministic=True,
                    verbose=0,
                )
                callbacks.append(eval_cb)

            callback = CallbackList(callbacks) if callbacks else None

            # config SB3 (factory)
            cfg = SB3Config(
                seed=pcfg.seed,
                device="cpu",
                verbose=1,
                policy="MlpPolicy",
                tensorboard_log=tb_dir,

                total_timesteps=int(hp.get("total_timesteps", 30_000)),
                learning_rate=float(hp.get("learning_rate", 3e-4)),
                gamma=float(hp.get("gamma", 0.99)),

                n_steps=int(hp.get("n_steps", 2048)),
                batch_size=int(hp.get("batch_size", 256)),
                n_epochs=int(hp.get("n_epochs", 10)),
                gae_lambda=float(hp.get("gae_lambda", 0.95)),
                clip_range=float(hp.get("clip_range", 0.2)),
                ent_coef=hp.get("ent_coef", 0.0),
                vf_coef=float(hp.get("vf_coef", 1.0)),
                max_grad_norm=float(hp.get("max_grad_norm", 0.5)),
                target_kl=hp.get("target_kl", None),

                buffer_size=int(hp.get("buffer_size", 300_000)),
                learning_starts=int(hp.get("learning_starts", 10_000)),
                train_freq=int(hp.get("train_freq", 1)),
                gradient_steps=int(hp.get("gradient_steps", 1)),
                tau=float(hp.get("tau", 0.005)),

                exploration_fraction=float(hp.get("exploration_fraction", 0.1)),
                exploration_final_eps=float(hp.get("exploration_final_eps", 0.02)),

                extra=hp.get("extra", None),
            )

            # build model + train
            model = model_choice(algo, cfg, train_env)
            model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

            # Eval: random / buy&hold / agent
            if needs_discrete:
                def random_act(_obs):
                    return np.array([eval_env.action_space.sample()])

                def buyhold_act(_obs):
                    return np.array([len(discrete_positions) - 1])

            else:
                def random_act(_obs):
                    return np.array([eval_env.action_space.sample()])

                def buyhold_act(_obs):
                    return np.array([[1.0]], dtype=np.float32)

            def agent_act(obs):
                a, _ = model.predict(obs, deterministic=True)
                return a

            rnd = eval_mean_final_portfolio(eval_env, random_act, pcfg.n_eval_episodes, pcfg.eval_seed_base + 1000)
            bh = eval_mean_final_portfolio(eval_env, buyhold_act, pcfg.n_eval_episodes, pcfg.eval_seed_base + 2000)
            ag = eval_mean_final_portfolio(eval_env, agent_act, pcfg.n_eval_episodes, pcfg.eval_seed_base + 3000)

            diff = (ag - bh) if np.isfinite(ag) and np.isfinite(bh) else float("nan")

            # log W&B
            wandb.log({
                "eval/random_mean_final": rnd,
                "eval/buyhold_mean_final": bh,
                "eval/agent_mean_final": ag,
                "eval/agent_minus_buyhold": diff,
            })
            wandb.run.summary["eval/agent_minus_buyhold"] = diff

            # save model + normalizer
            out_algo_dir = os.path.join(pcfg.out_dir, "bench_models", algo)
            Path(out_algo_dir).mkdir(parents=True, exist_ok=True)
            model_path = os.path.join(out_algo_dir, f"{algo}_{run.id}")
            norm_path = os.path.join(out_algo_dir, f"vecnormalize_{run.id}.pkl")

            model.save(model_path)
            try:
                train_env.save(norm_path)
            except Exception:
                pass

            wandb.run.summary["saved/model"] = model_path + ".zip"
            wandb.run.summary["saved/vecnormalize"] = norm_path

            # console
            print(f"[eval] random={rnd:.2f} | buyhold={bh:.2f} | agent={ag:.2f} | agent-bh={diff:.2f}")

            results.append({
                "algo": algo,
                "random_mean_final": rnd,
                "buyhold_mean_final": bh,
                "agent_mean_final": ag,
                "agent_minus_buyhold": diff,
            })

        except Exception as e:
            if run is not None:
                wandb.log({"error": f"{type(e).__name__}: {e}"})
            raise
        finally:
            try:
                if eval_env is not None:
                    eval_env.close()
            except Exception:
                pass
            try:
                if train_env is not None:
                    train_env.close()
            except Exception:
                pass
            try:
                if run is not None:
                    wandb.finish()
            except Exception:
                pass

    # ranking final (console + csv)
    results_sorted = sorted(
        results,
        key=lambda r: (r["agent_minus_buyhold"] if np.isfinite(r["agent_minus_buyhold"]) else -1e18),
        reverse=True,
    )

    print("\n===== RANKING (agent_minus_buyhold) =====")
    for r in results_sorted:
        print(
            f"{r['algo']:>12} | agent={r['agent_mean_final']:.2f} | "
            f"bh={r['buyhold_mean_final']:.2f} | diff={r['agent_minus_buyhold']:.2f}"
        )

    out_csv = "algo_benchmark_results.csv"
    pd.DataFrame(results_sorted).to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")


if __name__ == "__main__":
    main()
