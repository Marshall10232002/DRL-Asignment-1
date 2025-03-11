# Remember to adjust your student ID in meta.xml

import numpy as np
import pickle
import random
import os

# We do a simple lazy-load mechanism so we only read the file once
Q_VALUES = None  # Global dictionary for Q-values

def load_q_values():
    global Q_VALUES
    if Q_VALUES is None:
        if os.path.exists("q_values.pkl"):
            with open("q_values.pkl", "rb") as f:
                Q_VALUES = pickle.load(f)
        else:
            Q_VALUES = {}
    return Q_VALUES

def get_action(obs):
    """
    This function should:
      1) Load your Q-values from 'q_values.pkl' (if not already loaded).
      2) Convert `obs` to a proper key in that Q-value dictionary.
      3) Return the action that has the highest Q-value for that state if found.
         Otherwise, return a fallback action (e.g., random).
    """

    # 1) Load Q-values
    q_dict = load_q_values()

    # If your environment is standard Taxi-v3 with discrete states, `obs` is just an integer
    # If your environment is a 16-dim or some other representation, 
    #   you must create a consistent key from `obs`.
    # Example: key = obs if obs is an integer,
    #          or key = tuple(obs) if obs is multi-dimensional
    key = obs
    if not isinstance(key, int):
        # If your custom environment returns a tuple for state, do something like:
        key = tuple(obs)  # must be hashable

    # 2) Retrieve Q-values if they exist
    #if key in q_dict:
    #    q_vals = q_dict[key]
    #    # 3) Argmax
    #    best_action = int(np.argmax(q_vals))
    #    return best_action
    #else:
    #    # Fallback: random or a safe default
    #    return random.choice([0, 1, 2, 3, 4, 5])
    return random.choice([0, 1, 2, 3])
