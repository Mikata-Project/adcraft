"""Wrapper for flattening observations and actions of an environment."""
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from adcraft.gymnasium_kw_utils import flatten_dict_array


class FlatArrayWrapper(gym.Wrapper):
    """Observation wrapper that flattens the observation.

    Example usage:
        from ray.rllib.algorithms.sac import SACConfig
        from ray.tune.registry import register_env

        from adcraft.gymnasium_kw_env import BiddingSimulation
        from adcraft.wrappers.flat_array import FlatArrayWrapper


        register_env(
            "FlatArrayAuction",
            lambda x: FlatArrayWrapper(BiddingSimulation(config={}))
        )

        config = (
            SACConfig()
            .framework("torch")
            .rollouts(create_env_on_local_worker=True)
            .debugging(seed=0)
            .training(model={"fcnet_hiddens" : [32,32]})
            .environment(env = "FlatArrayAuction")
            .resources(num_gpus=1)
            .evaluation(
                evaluation_interval = 1
            )
        )

        sac = config.build()
        sac.train()
        sac.evaluate()
    """

    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)
        self.action_space = spaces.flatten_space(env.action_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten
        Returns:
            The flattened observation
        """
        return spaces.flatten(self.env.observation_space, observation)

    def action(self, action):
        """Flattens an action.

        Args:
            action: The action to flatten
        Returns:
            The flattened action
        """
        return spaces.unflatten(self.env.action_space, action)

    def step(self, action: spaces.Box) -> Tuple[spaces.Box, float, bool, bool, dict]:
        """Pass the action to the interal BiddingSimulation to get results."""
        observations, reward, terminated, truncated, info = self.env.step(
            spaces.unflatten(self.env.action_space, action)
        )
        flat_obs = flatten_dict_array(observations)
        return flat_obs, reward, terminated, truncated, info

    def reset(
        self, *args, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[spaces.Box, dict]:
        """Flatten overvations from reset."""
        observations, info = self.env.reset(*args, seed=seed, options=options)
        return self.observation(observations), info
