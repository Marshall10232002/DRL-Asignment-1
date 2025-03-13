import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!

class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=5000):
        """
        Custom Taxi environment supporting dynamic grid sizes and specifications.

        Parameters:
        - grid_size (int): Size of the grid (n x n), must be >= 5.
        - fuel_limit (int): Maximum fuel steps, defaults to 5000.
        """
        if grid_size < 5:
            raise ValueError("Grid size must be at least 5.")
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        self.stations = []  # To be set in reset
        self.passenger_loc = None
        self.obstacles = set()
        self.destination = None

    def reset(self):
        """
        Reset the environment with random configurations.

        Returns:
        - state (tuple): Initial state of the environment.
        - info (dict): Additional information (empty for now).
        """
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        # All possible positions on the grid
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

        # Randomly select 4 distinct positions for stations (R, G, Y, B)
        self.stations = random.sample(all_positions, 4)

        # Randomly choose passenger location from stations
        self.passenger_loc = random.choice(self.stations)

        # Randomly choose destination from remaining stations
        remaining_stations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(remaining_stations)

        # Place taxi randomly, avoiding station positions
        available_positions = [pos for pos in all_positions if pos not in self.stations]
        self.taxi_pos = random.choice(available_positions)

        # Add random obstacles (up to 20% of grid), avoiding stations and initial taxi position
        num_obstacles = random.randint(0, int(self.grid_size ** 2 * 0.2))
        available_for_obstacles = [pos for pos in all_positions 
                                  if pos not in self.stations and pos != self.taxi_pos]
        self.obstacles = set(random.sample(available_for_obstacles, 
                                           min(num_obstacles, len(available_for_obstacles))))

        return self.get_state(), {}

    def step(self, action):
        """
        Perform an action and update the environment state.

        Parameters:
        - action (int): Action to take (0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff).

        Returns:
        - state (tuple): New state after the action.
        - reward (float): Reward received from the action.
        - done (bool): Whether the episode has ended.
        - info (dict): Additional information (empty for now).
        """
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0

        # Movement actions
        if action == 0:  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2:  # Move East
            next_col += 1
        elif action == 3:  # Move West
            next_col -= 1

        if action in [0, 1, 2, 3]:  # Handle movement
            if (next_row, next_col) in self.obstacles or \
               not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5  # Hit obstacle or wall
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        elif action == 4:  # PICKUP
            if self.taxi_pos == self.passenger_loc and not self.passenger_picked_up:
                self.passenger_picked_up = True
            else:
                reward -= 10  # Incorrect pickup
        elif action == 5:  # DROPOFF
            if self.passenger_picked_up:
                if self.taxi_pos == self.destination:
                    reward += 50  # Successful dropoff
                    return self.get_state(), reward - 0.1, True, {}
                else:
                    reward -= 10  # Wrong dropoff location
                self.passenger_picked_up = False
                self.passenger_loc = self.taxi_pos
            else:
                reward -= 10  # No passenger to drop off

        reward -= 0.1  # Movement cost
        self.current_fuel -= 1

        # Check fuel depletion
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, {}  # Fuel depleted

        return self.get_state(), reward, False, {}

    def get_state(self):
        """
        Return the current environment state as a tuple.

        Returns:
        - state (tuple): A 16-element tuple containing:
          - Taxi row, column
          - Station positions (R, G, Y, B: row, column for each)
          - Obstacle indicators (north, south, east, west)
          - Passenger and destination proximity flags
        """
        taxi_row, taxi_col = self.taxi_pos

        # Obstacle indicators (1 if blocked, 0 if clear)
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col-1) in self.obstacles)

        # Passenger proximity (1 if passenger is in current or adjacent cell, 0 otherwise)
        passenger_look = int(any((taxi_row + dr, taxi_col + dc) == self.passenger_loc 
                                for dr, dc in [(0,0), (-1,0), (1,0), (0,1), (0,-1)] 
                                if 0 <= taxi_row + dr < self.grid_size and 0 <= taxi_col + dc < self.grid_size))

        # Destination proximity (1 if destination is in current or adjacent cell, 0 otherwise)
        destination_look = int(any((taxi_row + dr, taxi_col + dc) == self.destination 
                                  for dr, dc in [(0,0), (-1,0), (1,0), (0,1), (0,-1)] 
                                  if 0 <= taxi_row + dr < self.grid_size and 0 <= taxi_col + dc < self.grid_size))

        state = (taxi_row, taxi_col,
                 self.stations[0][0], self.stations[0][1],  # R
                 self.stations[1][0], self.stations[1][1],  # G
                 self.stations[2][0], self.stations[2][1],  # Y
                 self.stations[3][0], self.stations[3][1],  # B
                 obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                 passenger_look, destination_look)
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        """
        Render the current state of the environment to the console.

        Parameters:
        - taxi_pos (tuple): Current position of the taxi (row, col).
        - action (int, optional): Last action taken.
        - step (int, optional): Current step number.
        - fuel (int, optional): Remaining fuel.
        """
        clear_output(wait=True)
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place obstacles
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'

        # Place stations
        for i, station in enumerate(self.stations):
            grid[station[0]][station[1]] = ['R', 'G', 'Y', 'B'][i]

        # Place passenger if not picked up
        if not self.passenger_picked_up:
            py, px = self.passenger_loc
            grid[py][px] += 'P'

        # Place destination
        dy, dx = self.destination
        grid[dy][dx] += 'D'

        # Place taxi
        ty, tx = taxi_pos
        grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"Passenger: {'In Taxi' if self.passenger_picked_up else self.passenger_loc}")
        print(f"Destination: {self.destination}")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """
        Return a human-readable name for the action.

        Parameters:
        - action (int): Action index.

        Returns:
        - str: Action name.
        """
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"
    
def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    grid_size = random.randint(5, 10)
    env_config = {
        "grid_size": grid_size,
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")