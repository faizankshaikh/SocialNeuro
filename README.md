# Social Neuroscience Testbed

Welcome to the PettingZoo Gym Environment Walkthrough for SocialNeuro (2x2 foraging games). This guide will help you set up and run the environment step-by-step.

## Prerequisites

Before you begin, make sure you have the following prerequisites:

- Python environment
- Jupyter Notebook (optional)

## Installation

Install the required Python packages and clone the SocialNeuro repository:

```bash
pip install -q black gymnasium pettingzoo==1.22.3
git clone https://github.com/faizankshaikh/SocialNeuro
```

## Getting Started

Now that you have the necessary packages and the SocialNeuro repository installed, follow these steps to set up and run the PettingZoo Gym environment:

### Step 1: Navigate to the SocialNeuro Directory

Change your working directory to the SocialNeuro directory:

```bash
cd SocialNeuro
```

### Step 2: Running the Environment

Import the required libraries and create the SocialNeuro environment with the desired parameters:

```python
import json
import numpy as np
import pandas as pd
from SocialNeuro.envs.social_v0 import SocialNeuro

env = SocialNeuro(render_mode="human", cfg_name="comp_v0", reward_struct="egoistic")
```

Set the number of episodes and run the episodes while printing actions taken by players:

```python
episodes = 2

for episode in range(episodes):
    print(f"Episode #{episode+1}")
    print("=" * 10)
    obs = env.reset()
    print()

    while env.agents:
        acts = {
            "player1": np.random.choice([0, 1]),
            "player2": np.random.choice([0, 1])
        }
        print(f"--Action taken by player 1: {env.action_dict[acts['player1']]}")
        print(f"--Action taken by player 2: {env.action_dict[acts['player2']]}")
        print()

        obs, rews, terms, truncs, infos = env.step(acts)
        print()
```

This completes the setup and execution of the PettingZoo Gym environment for SocialNeuro. You can customize the environment parameters and actions according to your specific use case.

## How to create a new 2x2 game

Refer [Config creator notebook](https://github.com/faizankshaikh/SocialNeuro/blob/main/SocialNeuro/envs/cfg/config_creator.ipynb)

## Additional examples

If you prefer, you can write your own multi-agent RL algorithms. Otherwise, you can choose between

* [Stable-baselines](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html) for fully independent RL algorithms
* [Rlib](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#) for multi-agent compatible algorithms
* [Tianshou](https://github.com/thu-ml/tianshou) or [CleanRL](https://github.com/vwxyzjn/cleanrl) for simple research friendly implementations 

For more examples, you can refer to the experiments folder

Please feel free to explore and adapt the environment for your research or project needs. If you happen to have any issues or have questions, please don't hesitate to reach out to me for help.

Happy experimenting with PettingZoo and SocialNeuro!
