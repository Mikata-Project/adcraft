# AdCraft: An Advanced Reinforcement Learning Benchmark Environment for Search Engine Marketing Optimization
=======================

A Customizable Benchmark Environment for Reinforcement Learning Algorithms in Search Engine Marketing (SEM)

## Description

AdCraft is a Python-based framework designed as a customizable benchmark for evaluating the performance of reinforcement learning (RL) algorithms in the field of search engine marketing (SEM). This repository provides a comprehensive set of tools and resources, including a package of Gymnasium environments, to simulate SEM campaigns and assess the efficacy of RL algorithms across various auction scenarios.

## Features

- Built-in baseline algorithm for reference and performance comparison
- Seamless integration with popular RL algorithms: Proximal Policy Optimization (PPO), Advantage Actor-Critic (A2C), and Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Customizable environment parameters including keyword volume, click-through rate (CTR), and conversion rate (CVR)
- Evaluation metrics tailored for SEM, such as Average Keyword Normalized Cumulative Profit (AKNCP) and Normalized Conversion Profit (NCP)
- Support for studying environmental sparsity and non-stationarity, enabling exploration of real-world SEM dynamics
- Flexibility to implement and study non-stationary dynamics, reflecting real-world scenarios
- Package of Gymnasium environments for different auction scenarios


## Example usage with RLlib on GPU

[//]: # "TODO: Provide more examples on how to use without Ray."

```python
from adcraft.gymnasium_kw_env import BiddingSimulation
import ray
from ray.rllib.algorithms.ppo import PPOConfig

ray.init(
    num_gpus=1,
    ignore_reinit_error=True,
    log_to_driver=False,
    logging_level="WARNING",
    include_dashboard=False
)

config = (
    PPOConfig()
    .framework("torch")
    .rollouts(create_env_on_local_worker=True)
    .debugging(seed=0)
    .training(model={"fcnet_hiddens" : [32,32]}) #shape of policy observation to action
    .environment(env = BiddingSimulation)
    .resources(num_gpus=1)
)

ppo = config.build()

for _ in range(1):
    ppo.train()

eval_dict = ppo.evaluate()
eval_dict
```


## Installation

NOTE: To install you need to have Rust installed. Instructions are here: https://www.rust-lang.org/tools/install

Easiest method is to `pip install git+https://github.com/Mikata-Project/adcraft.git`

Alternate method is to clone repo and `make install`.


## Tests

Lint code with: `make lint`

Run tests with: `make test`

## License

 This project is licensed under the Apache-2.0 license.

 