import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pickle
from collections import deque
import os
from tqdm import tqdm

from simple_custom_taxi_env import SimpleTaxiEnv

# Set seeds for reproducibility if needed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Select device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# 1) Define the DQN Model
# ==============================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# 2) Replay Buffer
# ==============================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    env,
    num_episodes=5000,      # total number of episodes
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=5000,        # number of episodes over which epsilon decays
    target_update_freq=100,
    checkpoint_freq=500    # update q_values.pkl every these many episodes
):
    """
    Train a DQN on the given environment. The episode termination is controlled
    by the environment (e.g., fuel limit). Every checkpoint_freq episodes, the current
    Q-values are extracted and q_values.pkl is overwritten.
    """
    # -----------------------------------
    # 1) Setup networks, optimizer, etc.
    # -----------------------------------
    obs_sample, _ = env.reset()
    state_dim = len(obs_sample)
    action_dim = env.action_space.n

    dqn = DQN(state_dim, action_dim).to(device)
    target_dqn = DQN(state_dim, action_dim).to(device)
    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    # -----------------------------------
    # 2) Epsilon schedule: per-episode
    # -----------------------------------
    epsilon = eps_start
    if eps_decay > 0:
        eps_decrement = (eps_start - eps_end) / float(eps_decay)
    else:
        eps_decrement = 0.0

    global_step = 0  # counts steps across all episodes
    visited_states = set()
    reward_history = []  # for tracking progress

    # -----------------------------------
    # 3) Main training loop (episodes)
    # -----------------------------------
    for episode in tqdm(range(1, num_episodes + 1)):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32)
        visited_states.add(tuple(obs))
        episode_reward = 0

        # Run until the environment terminates the episode
        while True:
            global_step += 1

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = dqn(obs_tensor)
                action = q_vals.argmax(dim=1).item()

            # Step the environment
            next_obs, reward, done, truncated, _ = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32)
            visited_states.add(tuple(next_obs))
            replay_buffer.push(obs, action, reward, next_obs, float(done or truncated))

            obs = next_obs
            episode_reward += reward

            # Train if enough samples are available
            if len(replay_buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)

                states_t      = torch.tensor(states_b, dtype=torch.float32).to(device)
                actions_t     = torch.tensor(actions_b, dtype=torch.int64).unsqueeze(-1).to(device)
                rewards_t     = torch.tensor(rewards_b, dtype=torch.float32).to(device)
                next_states_t = torch.tensor(next_states_b, dtype=torch.float32).to(device)
                dones_t       = torch.tensor(dones_b, dtype=torch.float32).to(device)

                # Compute current Q-values
                current_q = dqn(states_t).gather(1, actions_t).squeeze(-1)

                # Compute target Q-values using the target network
                with torch.no_grad():
                    next_q = target_dqn(next_states_t).max(dim=1)[0]
                target_q = rewards_t + gamma * next_q * (1 - dones_t)

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network periodically
            if global_step % target_update_freq == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            # Terminate episode if the environment signals done
            if done or truncated:
                break

        # Decay epsilon after each episode
        if epsilon > eps_end:
            epsilon = max(eps_end, epsilon - eps_decrement)

        reward_history.append(episode_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episode {episode}/{num_episodes} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

        # Overwrite the q_values.pkl checkpoint periodically
        if episode % checkpoint_freq == 0:
            dqn.eval()
            q_dict = {}
            with torch.no_grad():
                for s in visited_states:
                    s_tensor = torch.tensor([s], dtype=torch.float32).to(device)
                    q_vals = dqn(s_tensor).squeeze(0).cpu().numpy()
                    q_dict[s] = q_vals
            with open("q_values.pkl", "wb") as f:
                pickle.dump(q_dict, f)
            dqn.train()
            print(f"Checkpoint updated: Q-values saved to 'q_values.pkl' at episode {episode}")

    # -----------------------------------
    # 4) Final extraction of Q-values & save to file
    # -----------------------------------
    dqn.eval()
    q_dict = {}
    with torch.no_grad():
        for s in visited_states:
            s_tensor = torch.tensor([s], dtype=torch.float32).to(device)
            q_vals = dqn(s_tensor).squeeze(0).cpu().numpy()
            q_dict[s] = q_vals

    with open("q_values.pkl", "wb") as f:
        pickle.dump(q_dict, f)
    print("Training completed, final Q-values saved to 'q_values.pkl'.")


if __name__ == "__main__":
    # Instantiate your environment with a fuel limit
    env_config = {"fuel_limit": 500}
    env = SimpleTaxiEnv(**env_config)

    # Start training with the updated configuration
    train_dqn(
        env,
        num_episodes=10000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=9800,
        target_update_freq=100,
        checkpoint_freq=500
    )
