import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from simple_custom_taxi_env import SimpleTaxiEnv  # make sure this matches your file name
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

# ---------------------------
# Print PyTorch and GPU Info
# ---------------------------
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# ---------------------------
# Hyperparameters (modifiable)
# ---------------------------
num_episodes = 1000         # Total episodes (used to approximate total timesteps)
avg_episode_length = 5000    # Estimated average episode length (adjust if needed)
total_timesteps = num_episodes * avg_episode_length

gamma = 0.9
learning_rate = 1.5e-2
replay_buffer_size = 10000
batch_size = 64
target_update_interval = 500   # in steps

# Exploration parameters:
exploration_initial_eps = 1.0
exploration_final_eps = 0.1
exploration_fraction = 0.8

# Checkpoint saving frequency (in episodes)
checkpoint_frequency = 1000  # every 1000 episodes

# Environment parameters
fuel_limit = 5000
min_grid_size = 5
max_grid_size = 5

# ---------------------------
# Gymnasium Wrapper for SimpleTaxiEnv
# ---------------------------
class TaxiEnvGymWrapper(gym.Env):
    """
    A Gymnasium wrapper for the custom taxi environment.
    Each episode randomizes the grid size between min_grid_size and max_grid_size.
    The observation is a 16-element vector where the first 10 elements (taxi and station coordinates)
    are normalized by the grid size.
    """
    def __init__(self, fuel_limit=5000, min_grid_size=5, max_grid_size=10):
        super(TaxiEnvGymWrapper, self).__init__()
        self.fuel_limit = fuel_limit
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.current_env = None
        self.current_grid_size = None

        # Define observation space:
        # - First 10 values (coordinates) are normalized to [0, 1]
        # - Last 6 values are binary flags (0 or 1)
        low = np.array([0.0]*10 + [0]*6, dtype=np.float32)
        high = np.array([1.0]*10 + [1]*6, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def reset(self, **kwargs):
        grid_size = random.randint(self.min_grid_size, self.max_grid_size)
        self.current_grid_size = grid_size
        self.current_env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=self.fuel_limit)
        state, _ = self.current_env.reset()
        state = self.normalize_state(state, grid_size)
        # Gymnasium reset() must return (observation, info)
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.current_env.step(action)
        obs = self.normalize_state(obs, self.current_grid_size)
        # Gymnasium step() must return (obs, reward, terminated, truncated, info)
        return np.array(obs, dtype=np.float32), reward, done, False, info

    def render(self, mode='human'):
        taxi_row, taxi_col, *_ = self.current_env.get_state()
        self.current_env.render_env((taxi_row, taxi_col))

    def normalize_state(self, state, grid_size):
        state = np.array(state, dtype=np.float32)
        state[:10] = state[:10] / grid_size
        return state

# ---------------------------
# Custom Callback for Logging Training Statistics
# ---------------------------
class TrainStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainStatsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # The Monitor wrapper adds an "episode" key in info when an episode finishes.
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                if self.episode_count % 100 == 0:
                    avg_reward = np.mean(self.episode_rewards[-100:])
                    avg_length = np.mean(self.episode_lengths[-100:])
                    print(f"Episode {self.episode_count} - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        return True


# ---------------------------
# Checkpoint Callback
# ---------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=checkpoint_frequency * avg_episode_length,  # approximate timesteps per checkpoint
    save_path='./',
    name_prefix='dqn_taxi_checkpoint'
)

if __name__ == "__main__":
    from stable_baselines3.common.vec_env import SubprocVecEnv
    def make_env():
        env = TaxiEnvGymWrapper(fuel_limit=fuel_limit, min_grid_size=min_grid_size, max_grid_size=max_grid_size)
        return Monitor(env)  # ✅ Wrap each individual environment

    # Create multiple environments wrapped with Monitor
    env = SubprocVecEnv([make_env for _ in range(1)])
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=replay_buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=(10, "step"),
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log="./dqn_taxi_tensorboard/"
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, TrainStatsCallback()])
    model.save("dqn_taxi_model")
    print("Training completed and model saved as dqn_taxi_model.zip")
