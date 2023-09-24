import numpy as np
import pandas as pd

from glob import glob
from gymnasium.spaces import Discrete, Box
from pettingzoo.utils.env import ParallelEnv

from SocialNeuro.envs.cfg.utils import _get_rewards_egoistic, _get_rewards_prosocial


class SocialNeuro(ParallelEnv):
    metadata = {"render_mode": ["human"]}

    def __init__(self, render_mode=None, cfg_name="social_v0", reward_struct="egoistic"):
        self.render_mode = render_mode

        self.game_init = pd.read_csv("SocialNeuro/envs/cfg/" + cfg_name + "_game_initialization.csv")
        self.game_mechanics = pd.read_csv("SocialNeuro/envs/cfg/" + cfg_name + "_game_mechanics.csv")

        self.reward_struct = reward_struct

        self.action_dict = {0: "wait", 1: "play", 2: "none"}
        self.num_actions = len(self.action_dict)

        self.possible_agents = ["player1", "player2"]
        self.agents = self.possible_agents[:]

        # set dummy values
        self.player1_prob_payoff = 0
        self.player2_prob_payoff = 0

        self.observation_spaces = {
            agent: Box(low=0, high=3, shape=(1, 5)) for agent in self.agents
        }

        self.action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.agents
        }

    def _get_obs(self):
        return {
            "player1": {
                "observation": np.array(
                    [
                        [
                            self.days_left,
                            self.player1_life_points,
                            self.player2_life_points,
                            self.player1_prob_payoff,
                            self.player2_prob_payoff,
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
                            self.player1_prob_payoff,
                            self.player2_prob_payoff,
                        ]
                    ]
                ),
                "action_mask": [1, 1, 0],
            },
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode == "human":
            self.render_text()

    def render_text(self, is_start=False, is_end=False):
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
        if is_init_val:
            # get days_left, life_points, individual and joint payoffs
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
        self.agents = self.possible_agents[:]

        self._set_game_var(is_init_val=True)

        self.player1_action = 2
        self.player2_action = 2
        self.num_life_points = self.game_init.player1_life_points.max()

        if self.reward_struct == "egoistic":
            self._get_rewards = _get_rewards_egoistic
        elif self.reward_struct == "prosocial":
            self._get_rewards = _get_rewards_prosocial

        if self.render_mode == "human":
            self.render_text(is_start=True)

        return self._get_obs()

    def step(self, actions):
        self.player1_action = actions["player1"]
        self.player2_action = actions["player2"]

        # reset payoffs according to actions
        self._set_game_var()

        # reset actions if dead    
        if self.player1_life_points == 0:
            self.player1_action = 2
        elif self.player2_life_points == 0:
            self.player2_action = 2

        player1_possible_outcome = np.random.uniform(0, 1) <= self.player1_prob_payoff
        player2_possible_outcome = np.random.uniform(0, 1) <= self.player2_prob_payoff

        player1_payoff, player2_payoff = self.game_mechanics[
            (self.game_mechanics.player1_alive == int(self.player1_life_points > 0))
            & (self.game_mechanics.player2_alive == int(self.player2_life_points > 0))
            & (self.game_mechanics.player1_action == self.player1_action)
            & (self.game_mechanics.player2_action == self.player2_action)
            & (self.game_mechanics.player1_possible_outcome == player1_possible_outcome)
            & (self.game_mechanics.player2_possible_outcome == player2_possible_outcome)
        ].values[0][-2:]

        self.player1_life_points += player1_payoff
        self.player1_life_points = np.clip(
            self.player1_life_points, 0, self.num_life_points - 1
        )
        self.player2_life_points += player2_payoff
        self.player2_life_points = np.clip(
            self.player2_life_points, 0, self.num_life_points - 1
        )

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

        self.days_left -= 1
        self._set_game_var(new_weather=True)

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

            if self.render_mode == "human":
                self.render_text(is_end=True)

            return (
                self._get_obs(),
                self._get_rewards(self.player1_life_points > 0, self.player2_life_points > 0),
                terminations,
                truncations,
                infos,
            )

        if self.render_mode == "human":
            self.render_text()

        return self._get_obs(), rewards, terminations, truncations, infos

