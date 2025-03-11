import gym
from gym import spaces
import numpy as np
import random
import time
from IPython.display import clear_output

class SimpleTaxiEnv(gym.Env):
    """
    Custom Taxi Environment for Assignment Q4.

    - Dynamic Grid: n x n where n is a random integer between 5 and 10.
    - Stations (R, G, Y, B) are placed randomly each episode.
    - Passenger pickup (P) is chosen randomly from the stations;
      the destination (D) is then chosen randomly from the remaining stations.
    - Approximately 20% of non‑station cells are randomly set as obstacles (X).
      Moving into an obstacle or off the grid incurs a penalty (-5).
    - Game Rules:
         • Movement (actions 0-3) cost -0.1 point.
         • Incorrect PICKUP (action 4) or DROPOFF (action 5) incur -10 points.
         • Successful dropoff awards +50 points and ends the episode.
         • Fuel is decremented each step; if fuel reaches 0, the episode ends with a -10 penalty.
    - Observation (16 integers):
         (taxi_row, taxi_col, st0_row, st0_col, st1_row, st1_col, 
          st2_row, st2_col, st3_row, st3_col, obstacle_north, obstacle_south, 
          obstacle_east, obstacle_west, passenger_flag, destination_flag)
    """
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, fuel_limit=5000):
        super(SimpleTaxiEnv, self).__init__()
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False

        # These will be set in reset():
        self.grid_size = None
        self.stations = None            # List of 4 station positions (R, G, Y, B)
        self.passenger_station = None     # Passenger pickup location (one of the stations)
        self.destination_station = None   # Destination (another station)
        self.obstacles = None           # Set of obstacle positions (X)
        self.taxi_pos = None

        # Define action space: 6 discrete actions (0: South, 1: North, 2: East, 3: West, 4: PICKUP, 5: DROPOFF)
        self.action_space = spaces.Discrete(6)

        # Observation: 16-tuple of integers.
        # The upper bounds here are set loosely.
        self.observation_space = spaces.Tuple([spaces.Discrete(100)] * 16)

        # Station labels for rendering.
        self.station_labels = ['R', 'G', 'Y', 'B']

    def reset(self):
        # Set grid size randomly between 5 and 10.
        self.grid_size = random.randint(5, 10)
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        # All positions in the grid.
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

        # Randomly choose 4 unique positions for stations.
        self.stations = random.sample(all_positions, 4)

        # Randomly choose passenger pickup station (P) from the stations.
        self.passenger_station = random.choice(self.stations)
        # Destination is chosen from the remaining stations.
        remaining = [pos for pos in self.stations if pos != self.passenger_station]
        self.destination_station = random.choice(remaining)

        # Place obstacles in ~20% of free cells (cells not occupied by stations).
        free_positions = [pos for pos in all_positions if pos not in self.stations]
        num_obstacles = int(0.2 * len(free_positions))
        if num_obstacles > 0:
            self.obstacles = set(random.sample(free_positions, num_obstacles))
        else:
            self.obstacles = set()

        # Choose taxi starting position from free positions that are not obstacles.
        taxi_candidates = [pos for pos in free_positions if pos not in self.obstacles]
        if taxi_candidates:
            self.taxi_pos = random.choice(taxi_candidates)
        else:
            self.taxi_pos = random.choice(all_positions)

        return self.get_state(), {}

    def get_state(self):
        taxi_r, taxi_c = self.taxi_pos
        # Unpack station positions.
        st0 = self.stations[0]
        st1 = self.stations[1]
        st2 = self.stations[2]
        st3 = self.stations[3]

        # Compute obstacle indicators: 1 if adjacent cell is off-grid or an obstacle.
        obstacle_north = 1 if taxi_r == 0 or ((taxi_r - 1, taxi_c) in self.obstacles) else 0
        obstacle_south = 1 if taxi_r == self.grid_size - 1 or ((taxi_r + 1, taxi_c) in self.obstacles) else 0
        obstacle_east  = 1 if taxi_c == self.grid_size - 1 or ((taxi_r, taxi_c + 1) in self.obstacles) else 0
        obstacle_west  = 1 if taxi_c == 0 or ((taxi_r, taxi_c - 1) in self.obstacles) else 0

        # Passenger flag: 1 if taxi is at the passenger station and passenger not picked up.
        passenger_flag = 1 if (self.taxi_pos == self.passenger_station and not self.passenger_picked_up) else 0
        # Destination flag: 1 if taxi is at the destination and passenger is onboard.
        destination_flag = 1 if (self.taxi_pos == self.destination_station and self.passenger_picked_up) else 0

        state = (taxi_r, taxi_c,
                 st0[0], st0[1],
                 st1[0], st1[1],
                 st2[0], st2[1],
                 st3[0], st3[1],
                 obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                 passenger_flag, destination_flag)
        return state

    def step(self, action):
        reward = 0
        done = False

        # Movement actions: 0: South, 1: North, 2: East, 3: West.
        if action in [0, 1, 2, 3]:
            taxi_r, taxi_c = self.taxi_pos
            new_r, new_c = taxi_r, taxi_c
            if action == 0:  # Move South
                new_r += 1
            elif action == 1:  # Move North
                new_r -= 1
            elif action == 2:  # Move East
                new_c += 1
            elif action == 3:  # Move West
                new_c -= 1

            # Check boundaries.
            if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
                reward -= 5  # Penalty for moving out of bounds.
            # Check for obstacles.
            elif (new_r, new_c) in self.obstacles:
                reward -= 5  # Penalty for hitting an obstacle.
            else:
                self.taxi_pos = (new_r, new_c)
                reward -= 0.1  # Movement cost.

        # PICKUP action (4).
        elif action == 4:
            if self.taxi_pos == self.passenger_station and not self.passenger_picked_up:
                self.passenger_picked_up = True
            else:
                reward -= 10  # Incorrect pickup.

        # DROPOFF action (5).
        elif action == 5:
            if self.taxi_pos == self.destination_station and self.passenger_picked_up:
                reward += 50
                done = True  # Successful dropoff ends the episode.
            else:
                reward -= 10  # Incorrect dropoff.
                # Optionally reset passenger state.
                self.passenger_picked_up = False

        # Deduct fuel cost.
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            done = True
            reward -= 10  # Penalty for fuel depletion.

        return self.get_state(), reward, done,False, {}

    def render(self, mode="ansi"):
        # Create an empty grid.
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        # Place obstacles.
        for (r, c) in self.obstacles:
            grid[r][c] = "X"
        # Place stations with their labels.
        for idx, pos in enumerate(self.stations):
            r, c = pos
            grid[r][c] = self.station_labels[idx]
        # Mark passenger pickup location with "P" if not picked up.
        if not self.passenger_picked_up:
            pr, pc = self.passenger_station
            grid[pr][pc] = "P"
        # Mark destination with "D".
        dr, dc = self.destination_station
        grid[dr][dc] = "D"
        # Place the taxi.
        tr, tc = self.taxi_pos
        grid[tr][tc] = "🚖"
        return "\n".join(" ".join(cell for cell in row) for row in grid)

    def render_env(self, action=None, step=None):
        clear_output(wait=True)
        print(f"Step: {step}")
        print(f"Fuel Left: {self.current_fuel}")
        print(f"Last Action: {action}")
        print(self.render())
        print()

def run_agent(agent_file, env_config, render=False):
    # Dynamically load the student agent module (must provide get_action(obs) function).
    import importlib.util
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    if render:
        env.render_env(action=None, step=step_count)
        time.sleep(0.5)

    while not done:
        action = student_agent.get_action(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1
        if render:
            env.render_env(action=action, step=step_count)
            time.sleep(0.2)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {"fuel_limit": 5000}
    # When run, the environment will have a random grid size (n between 5 and 10),
    # randomized station positions, obstacles, and passenger/destination assignment.
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")
