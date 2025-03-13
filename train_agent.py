import random
import numpy as np
import pickle
from  complete_env import SimpleTaxiEnv
from tqdm import tqdm

# Hyperparameters
NUM_EPISODES = 5000
N_STEPS = 2         # n-step return horizon
ALPHA = 0.15         # Learning rate
GAMMA = 0.9        # Discount factor
EPSILON = 1.0       # Starting exploration rate
EPSILON_MIN = 0.05  # Minimum exploration rate
EPSILON_DECAY = 0.9993  # Decay factor per episode

# Q-table: keys are states (tuples) and values are np.arrays of length 6 (one per action)
Q_table = {}

def get_Q(state):
    """Return Q-values for a state, initializing if needed."""
    if state not in Q_table:
        Q_table[state] = np.zeros(6)
    return Q_table[state]

def choose_action(state, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.choice(range(6))
    else:
        return int(np.argmax(get_Q(state)))

def train():
    global EPSILON
    env = SimpleTaxiEnv(fuel_limit=5000)
    total_rewards = []

    for episode in tqdm(range(NUM_EPISODES)):
        state, _ = env.reset()
        buffer = []  # To store transitions: each element is (state, action, reward)
        done = False
        step_count = 0
        episode_reward = 0

        while not done:
            action = choose_action(state, EPSILON)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            buffer.append((state, action, reward))
            
            # When enough transitions have been collected, perform an n-step update for the oldest transition
            if len(buffer) >= N_STEPS:
                G = 0
                for i in range(N_STEPS):
                    G += (GAMMA ** i) * buffer[i][2]
                # If the episode is not finished, add the estimated future reward from the state reached after n steps
                if not done:
                    G += (GAMMA ** N_STEPS) * np.max(get_Q(next_state))
                s0, a0, _ = buffer[0]
                Q_old = get_Q(s0)[a0]
                Q_table[s0][a0] = Q_old + ALPHA * (G - Q_old)
                # Remove the oldest transition from the buffer
                buffer.pop(0)
            
            state = next_state
            step_count += 1
            
            if done:
                # Flush remaining transitions in the buffer using a shorter horizon
                for i in range(len(buffer)):
                    G = 0
                    for j in range(len(buffer) - i):
                        G += (GAMMA ** j) * buffer[i+j][2]
                    s_i, a_i, _ = buffer[i]
                    Q_old = get_Q(s_i)[a_i]
                    Q_table[s_i][a_i] = Q_old + ALPHA * (G - Q_old)
                break
        
        total_rewards.append(episode_reward)
        # Decay the exploration rate
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(EPSILON, EPSILON_MIN)
        
        # Print progress and save a checkpoint every 100 episodes
        if episode % 500 == 0:
            avg_reward = np.mean(total_rewards[-500:])  # Average of the last 100 episodes
            print(f"Episode {episode}: Steps: {step_count}, Avg Reward: {avg_reward:.2f}, Epsilon: {EPSILON:.4f}")
            
    # Final Q-table save
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)
    print("Training finished. Final Q-table saved as q_table.pkl")

if __name__ == '__main__':
    train()

