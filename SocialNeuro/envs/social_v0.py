
import numpy as np
import pandas as pd

from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box


class SocialNeuro(ParallelEnv):
    metadata = {"render_mode": ["human"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.game_init = pd.read_csv("envs/cfg/social_v0_game_initialization.csv")
        self.game_mechanics = pd.read_csv("envs/cfg/social_v0_game_mechanics.csv")

        # bug
        self.num_life_points = 4

        self.action_dict = {0: "wait", 1: "play", 2: "none"}
        self.num_actions = len(self.action_dict)

        self.possible_agents = ["player1", "player2"]
        self.agents = self.possible_agents[:]

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

    def render_text(self, is_start=False):
        print(f"--Days left: {self.days_left}")
        print(f"--Current life of agent 1: {self.player1_life_points}")
        print(f"--Current life of agent 2: {self.player2_life_points}")
        print(f"--Probability of payoff for agent 1: {self.player1_prob_payoff}")
        print(f"--Probability of payoff for agent 2: {self.player2_prob_payoff}")

        if not is_start:
            print(
                f"--Previous action of agent 1: {self.action_dict[self.player1_action]}"
            )
            print(
                f"--Previous action of agent 2: {self.action_dict[self.player2_action]}"
            )

    def _get_rewards(self):
        return {
            "player1": -1 if self.player1_life_points == 0 else 0,
            "player2": -1 if self.player2_life_points == 0 else 0,
        }

    def _set_game_var(self, is_init_val=False):
        if is_init_val:
            (
                _,
                self.days_left,
                self.player1_life_points,
                self.player2_life_points,
                _,
                _,
                self.player1_prob_payoff,
                self.player2_prob_payoff,
            ) = (
                self.game_init[self.game_init.is_init_val == int(is_init_val)]
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
                self.player1_prob_payoff,
                self.player2_prob_payoff,
            ) = (
                self.game_init[
                    (self.game_init.days_left == self.days_left)
                    & (self.game_init.player1_life_points == self.player1_life_points)
                    & (self.game_init.player1_life_points == self.player1_life_points)
                ]
                .sample(1)
                .values[0]
            )

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        self._set_game_var(is_init_val=True)

        self.player1_action = 2
        self.player2_action = 2

        if self.render_mode == "human":
            self.render_text(is_start=True)

        return self._get_obs()

    def step(self, actions):
        self.player1_action = actions["player1"]
        self.player2_action = actions["player2"]

        if (self.player1_life_points != 0 and self.player2_life_points != 0) and (self.player1_action == self.player2_action == 1):
            # reset for special case when they both cooperate
            self._set_game_var()
        elif self.player1_life_points == 0:
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
        infos = {a: {} for a in self.agents}

        self.days_left -= 1

        if self.render_mode == "human":
            self.render_text()

        if self.days_left == 0 or (
            self.player1_life_points == 0 and self.player2_life_points == 0
        ):
            truncations = {a: True for a in self.agents}
            terminations = {a: False for a in self.agents}
            infos = {a: {} for a in self.agents}

            self.agents = []
            return (
                self._get_obs(),
                self._get_rewards(),
                terminations,
                truncations,
                infos,
            )

        self._set_game_var()

        return self._get_obs(), rewards, terminations, truncations, infos

