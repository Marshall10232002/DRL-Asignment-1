import numpy as np
import random
import torch
from stable_baselines3 import DQN

# Use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Load the trained model. (Ensure "dqn_taxi_model.zip" exists in the working directory.)
    model = DQN.load("dqn_taxi_model", device=device)
except Exception as e:
    print("Model not found or error in loading:", e)
    model = None

def get_action(obs):
    """
    Returns an action based on the trained DQN model.
    
    The observation is expected to be a 16-element array. If the first 10 values (coordinates)
    are not normalized (i.e. have values greater than 1), they are normalized by dividing by the inferred grid size,
    which is determined as (max(first 10 values) + 1). If any error occurs, a random action is returned.
    """
    try:
        obs = np.array(obs, dtype=np.float32)
        # Normalize if necessary.
        if np.max(obs[:10]) > 1.0:
            grid_size = int(np.max(obs[:10])) + 1
            obs[:10] = obs[:10] / grid_size
        # The Stable Baselines3 model expects a 2D array.
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    except Exception as ex:
        return random.choice([0, 1, 2, 3])
