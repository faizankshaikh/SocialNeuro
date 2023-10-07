# import libraries and modules
import numpy as np
import pandas as pd

from glob import glob
from gymnasium.spaces import Discrete, Box
from pettingzoo.utils.env import ParallelEnv

from SocialNeuro.envs.cfg.utils import _get_rewards_egoistic, _get_rewards_prosocial

class SocialNeuro(ParallelEnv):
    metadata = {"render_mode": ["human"]}

    def __init__(self, render_mode=None, cfg_name="social_v0", reward_struct="egoistic"):
        # Initialize the SocialNeuro environment with optional parameters
        self.render_mode = render_mode

        # Load game initialization and mechanics data from CSV files based on the provided cfg_name
        self.game_init = pd.read_csv("SocialNeuro/envs/cfg/" + cfg_name + "_game_initialization.csv")
        self.game_mechanics = pd.read_csv("SocialNeuro/envs/cfg/" + cfg_name + "_game_mechanics.csv")

        # Set the reward structure (egoistic or prosocial)
        self.reward_struct = reward_struct

        # Define action mappings and action spaces
        self.action_dict = {0: "wait", 1: "play", 2: "none"}
        self.num_actions = len(self.action_dict)

        # Define possible agents and initialize observation and action spaces for each agent
        self.possible_agents = ["player1", "player2"]
        self.agents = self.possible_agents[:]

        # Initialize dummy values for variables
        self.player1_prob_payoff = 0
        self.player2_prob_payoff = 0

        # Define observation and action spaces for each agent
        self.observation_spaces = {
            agent: Box(low=0, high=3, shape=(1, 6)) for agent in self.agents
        }
        self.action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.agents
        }

    def _get_obs(self):
        # Return observations for each agent including game state information
        return {
            "player1": {
                "observation": np.array(
                    [
                        [
                            self.days_left,
                            self.player1_life_points,
                            self.player2_life_points,
                            self.indiv_prob_payoff,
                            self.joint_prob_payoff,
                            self.diff_prob_payoff
                        ]
                    ]
                ),
                "action_mask": [1, 1, 0],
            },
            "player2": {
                "observation": np.array(
                    [
                        [
                            self.days_left,
                            self.player1_life_points,
                            self.player2_life_points,
                            self.indiv_prob_payoff,
                            self.joint_prob_payoff,
                            self.diff_prob_payoff
                        ]
                    ]
                ),
                "action_mask": [1, 1, 0],
            },
        }

    def observation_space(self, agent):
        # Get the observation space for a specific agent
        return self.observation_spaces[agent]

    def action_space(self, agent):
        # Get the action space for a specific agent
        return self.action_spaces[agent]

    def render(self):
        # Render the environment, typically in human-readable form
        if self.render_mode == "human":
            self.render_text()

    def render_text(self, is_start=False, is_end=False):
        # Render environment information in text format
        print(f"--Days left: {self.days_left}")
        print(f"--Current life of agent 1: {self.player1_life_points}")
        print(f"--Current life of agent 2: {self.player2_life_points}")

        if not is_start:
            print(
                f"--Previous action of agent 1: {self.action_dict[self.player1_action]}"
            )
            print(
                f"--Previous action of agent 2: {self.action_dict[self.player2_action]}"
            )

        if not is_end:
            print(f"--Individual Probability of payoff: {self.indiv_prob_payoff}")
            print(f"--Joint Probability of payoff: {self.joint_prob_payoff}")
            print(f"--Difference in Probability of payoff from the other weather: {self.diff_prob_payoff}")
            print(f"--Forest Index: {self.forest_index}")
            print(f"--Weather Index: {self.weather_index}")

    def _set_game_var(self, is_init_val=False, new_weather=False):
        # Set game variables based on conditions (init value, new weather, or default)
        # These variables determine the game state
        if is_init_val:
            # Initialize game state variables from the game_init data
            (
                _,
                self.days_left,
                self.player1_life_points,
                self.player2_life_points,
                _,
                _,
                self.forest_index,
                self.weather_index,
                self.indiv_prob_payoff,
                self.joint_prob_payoff,
                self.diff_prob_payoff,
                _,
                _,
            ) = (
                self.game_init[
                (self.game_init.is_init_val == 1)
                ]
                .sample(1)
                .values[0]
            )
        elif new_weather:
            # Update game state variables when transitioning to a new weather condition
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                self.weather_index,
                self.indiv_prob_payoff,
                self.joint_prob_payoff,
                self.diff_prob_payoff,
                _,
                _,
            ) = (
                self.game_init[
                (self.game_init.days_left == self.days_left)
                & (self.game_init.player1_life_points == self.player1_life_points)
                & (self.game_init.player2_life_points == self.player2_life_points)
                & (self.game_init.forest_index == self.forest_index)
                & (self.game_init.weather_index == np.random.choice([0, 1]))
                ]
                .sample(1)
                .values[0]
            )
        else:
            # Update game state variables based on agent actions and other conditions
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                self.player1_prob_payoff,
                self.player2_prob_payoff,
            ) = (
                self.game_init[
                    (self.game_init.days_left == self.days_left)
                    & (self.game_init.player1_life_points == self.player1_life_points)
                    & (self.game_init.player2_life_points == self.player2_life_points)
                    & (self.game_init.player1_action == self.player1_action)
                    & (self.game_init.player2_action == self.player2_action)
                    & (self.game_init.forest_index == self.forest_index)
                    & (self.game_init.weather_index == self.weather_index)
                ]
                .values[0]
            )

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state
        self.agents = self.possible_agents[:]

        # Initialize game variables and actions
        self._set_game_var(is_init_val=True)
        self.player1_action = 2
        self.player2_action = 2
        self.num_life_points = self.game_init.player1_life_points.max()

        # Set the reward calculation function based on the reward structure
        if self.reward_struct == "egoistic":
            self._get_rewards = _get_rewards_egoistic
        elif self.reward_struct == "prosocial":
            self._get_rewards = _get_rewards_prosocial

        # Render the initial state if in human render mode
        if self.render_mode == "human":
            self.render_text(is_start=True)

        # Return initial observations
        return self._get_obs()

    def step(self, actions):
        # Take a step in the environment based on agent actions
        self.player1_action = actions["player1"]
        self.player2_action = actions["player2"]

        # Update payoffs and game state variables based on actions
        self._set_game_var()

        # Reset actions if an agent is dead
        if self.player1_life_points == 0:
            self.player1_action = 2
        elif self.player2_life_points == 0:
            self.player2_action = 2

        # Determine possible outcomes based on probabilities
        player1_possible_outcome = np.random.uniform(0, 1) <= self.player1_prob_payoff
        player2_possible_outcome = np.random.uniform(0, 1) <= self.player2_prob_payoff

        # Get payoffs for each player based on the game mechanics
        player1_payoff, player2_payoff = self.game_mechanics[
            (self.game_mechanics.player1_alive == int(self.player1_life_points > 0))
            & (self.game_mechanics.player2_alive == int(self.player2_life_points > 0))
            & (self.game_mechanics.player1_action == self.player1_action)
            & (self.game_mechanics.player2_action == self.player2_action)
            & (self.game_mechanics.player1_possible_outcome == player1_possible_outcome)
            & (self.game_mechanics.player2_possible_outcome == player2_possible_outcome)
        ].values[0][-2:]

        # Update player life points and clip within valid range
        self.player1_life_points += player1_payoff
        self.player1_life_points = np.clip(
            self.player1_life_points, 0, self.num_life_points - 1
        )
        self.player2_life_points += player2_payoff
        self.player2_life_points = np.clip(
            self.player2_life_points, 0, self.num_life_points - 1
        )

        # Initialize rewards, terminations, truncations, and infos
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {
            "player1": {
                "prob_payoff": self.player1_prob_payoff
            },
            "player2": {
                "prob_payoff": self.player2_prob_payoff
            }
        }

        # Update the number of days left and set new weather conditions
        self.days_left -= 1
        self._set_game_var(new_weather=True)

        # Handle termination conditions
        if self.days_left == 0 or (
            self.player1_life_points == 0 and self.player2_life_points == 0
        ):
            truncations = {a: True for a in self.agents}
            terminations = {a: False for a in self.agents}
            infos = {
                "player1": {
                    "prob_payoff": self.player1_prob_payoff
                },
                "player2": {
                    "prob_payoff": self.player2_prob_payoff
                }
            }

            self.agents = []

            # Render the end state if in human render mode
            if self.render_mode == "human":
                self.render_text(is_end=True)

            return (
                self._get_obs(),
                self._get_rewards(self.player1_life_points > 0, self.player2_life_points > 0),
                terminations,
                truncations,
                infos,
            )

        # Render the current state if in human render mode
        if self.render_mode == "human":
            self.render_text()

        # Return new observations, rewards, terminations, truncations, and infos
        return self._get_obs(), rewards, terminations, truncations, infos
