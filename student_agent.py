import pickle
import numpy as np
import random

# Load the pre-trained Q-table from disk.
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    print("Q-table not found. Falling back to random actions.")
    q_table = {}

def get_action(obs):
    """
    Returns an action based on the trained Q-table.
    If the observation is unseen, returns a random action.
    """
    if obs not in q_table:
        print("random")
        return random.choice([0, 1])
    else:
        # Use a greedy policy: choose the action with the highest Q-value.
        return int(np.argmax(q_table[obs]))

