import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from gym_trading_env.wrapper import DiscreteActionsWrapper
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1_000_000,
    "learning_rate": 1e-4,
    "ent_coef": 0.05,
    "batch_size": 1024,
    "n_steps": 4096,
    "window_size": 40,
    "positions": [-1, 0, 1],
    "project_name": "RL-Trading-Project",
    "run_name": "PPO_Window40_ZScore_Fix"
}


def calculate_indicators(df):
    # 1. Calcul des indicateurs techniques
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_ret'].rolling(window=14).std()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    sma = df['close'].rolling(window=20).mean()
    df['dist_sma'] = (df['close'] - sma) / sma

    # 2. Normalisation Z-SCORE
    cols_to_norm = ['log_ret', 'volatility', 'rsi', 'dist_sma']
    for col in cols_to_norm:
        mean = df[col].rolling(100).mean()
        std = df[col].rolling(100).std()
        # On ajoute le préfixe 'z_'
        df[f'z_{col}'] = (df[col] - mean) / (std + 1e-8)

    cols_to_keep = ['close', 'high', 'low', 'open'] + [c for c in df.columns if c.startswith('z_')]

    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    return df[cols_to_keep].dropna()


def preprocess(df):
    df = df.sort_index().dropna().drop_duplicates()
    return calculate_indicators(df)


def reward_function(history):
    curr_val = history['portfolio_valuation', -1]
    prev_val = history['portfolio_valuation', -2]
    agent_ret = np.log(curr_val / prev_val)

    curr_price = history['data_close', -1]
    prev_price = history['data_close', -2]
    market_ret = np.log(curr_price / prev_price)

    reward = agent_ret - market_ret
    if abs(reward) < 1e-6:
        reward -= 0.001

    return reward * 100


run = wandb.init(
    project=config["project_name"],
    name=config["run_name"],
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)


def make_env():
    # Env de base
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir="./data/*.pkl",
        preprocess=preprocess,
        portfolio_initial_value=1000,
        trading_fees=0.1 / 100,
        borrow_interest_rate=0.02 / 100 / 24,
        reward_function=reward_function,
    )
    # Actions discrètes
    env = DiscreteActionsWrapper(env, positions=config["positions"])
    # Aplatissement pour le MLP
    env = FlattenObservation(env)
    return env


env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=config["window_size"])

print(f"--- Démarrage Phase 4 Fix ---")
print(f"Window Size: {config['window_size']}")

model = PPO(
    config["policy_type"],
    env,
    verbose=1,
    learning_rate=config["learning_rate"],
    ent_coef=config["ent_coef"],
    batch_size=config["batch_size"],
    n_steps=config["n_steps"],
    tensorboard_log=f"runs/{run.id}"
)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
)

model.save("ppo_windowed_fix")

# --- 7. ÉVALUATION ---
print("Évaluation...")
obs = env.reset()
done = False


eval_env = make_env()
eval_env = DiscreteActionsWrapper(
    gym.make("MultiDatasetTradingEnv", dataset_dir="./data/*.pkl", preprocess=preprocess, portfolio_initial_value=1000,
             trading_fees=0.1 / 100, borrow_interest_rate=0.02 / 100 / 24, reward_function=reward_function),
    positions=config["positions"]
)

wandb.finish()