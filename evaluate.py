import numpy as np
import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO

# ==============================
# CONFIG (same as training)
# ==============================

NET_FILE = "net.net.xml"
ROUTE_FILE = "routes.rou.xml"

USE_GUI = True      # 👈 IMPORTANT (to see simulation)
DELTA_TIME = 5
YELLOW_TIME = 3
NUM_SECONDS = 1800

MODEL_PATH = "ppo_traffic_model.zip"

# ==============================
# WRAPPER (same as training)
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
# MAIN EVALUATION
# ==============================

def main():

    print("🚦 Loading environment with GUI...")

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

    env = SumoSingleAgentWrapper(env)

    print("🤖 Loading trained model...")
    model = PPO.load(MODEL_PATH)

    obs, _ = env.reset()
    done = False

    print("🚀 Running evaluation...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("✅ Simulation finished.")

    env.close()


if __name__ == "__main__":
    main()
