import numpy as np
import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback  # ✅ ADDED

# ==============================
# CONFIG
# ==============================

NET_FILE = "net.net.xml"
ROUTE_FILE = "routes.rou.xml"

USE_GUI = True
DELTA_TIME = 5
YELLOW_TIME = 3
NUM_SECONDS = 1800
TRAIN_STEPS = 400000

# ==============================
# WRAPPER (FIXED)
# ==============================

class SumoSingleAgentWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed=None, options=None):
        obs = self.env.reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        return np.array(obs, dtype=np.float32), {}

    def step(self, action):

        result = self.env.step(int(action))

        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        reward = reward / 100.0

        return (
            np.array(obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.env.close()

# ==============================
# ENV CREATION
# ==============================

def make_env():
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=USE_GUI,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        reward_fn="pressure",
        single_agent=True
    )

    print("🎯 Single-agent traffic control active")

    env = SumoSingleAgentWrapper(env)
    env = Monitor(env)

    return env

# ==============================
# MAIN
# ==============================

def main():

    print("🚦 Creating environment...")

    env = DummyVecEnv([make_env])

    print("✅ Environment ready")

    # ✅ CHECKPOINT CALLBACK (saves every 10,000 steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="ppo_traffic"
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,   # ✅ comma fixed
        clip_range=0.2,
        verbose=1,
        device="cpu"
    )

    print("🚀 Training started...")

    model.learn(
        total_timesteps=TRAIN_STEPS,
        callback=checkpoint_callback
    )

    model.save("ppo_traffic_model")

    print("🎉 Training complete!")

    env.close()


if __name__ == "__main__":
    main()
