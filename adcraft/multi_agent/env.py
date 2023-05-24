"""Functions for Multi-Agent environments."""
from ray.rllib.env.multi_agent_env import make_multi_agent, MultiAgentEnv

from adcraft.gymnasium_kw_env import BiddingSimulation
from adcraft.wrappers.flat_array import FlatArrayWrapper


def make_multi_flat(num_agents: int) -> MultiAgentEnv:
    """
    Make multiple simulations in one class.

    Usage:
        env = make_multi_flat(3)
        obs, info = env.reset()
        print(obs)
        >  {0: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                ),
            1: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                ),
            2: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                )
            }
    """
    ma_flat_cls = make_multi_agent(
        lambda x: FlatArrayWrapper(BiddingSimulation(config={}))
    )
    ma_flat = ma_flat_cls({"num_agents": num_agents})

    return ma_flat
