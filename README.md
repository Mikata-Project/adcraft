adcraft
=======================

Gym environment simulating auctions.


## Description

Package of Gymnasium environments for different auction types and styles.
Typically these environments will be used for Reinforcement Learning
algorithms.


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


## Tests

Lint code with

::

    $ make lint


Run tests with

::

    $ make test


## Install

Install without running linter and unit tests:

::

    $ make install


Run linter and unit tests:

::

    $ make
