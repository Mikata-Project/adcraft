import pytest

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from adcraft.multi_agent.env import make_multi_flat


@pytest.mark.unit
def test_make_multi_flat_type() -> None:
    env = make_multi_flat(2)
    assert isinstance(env, MultiAgentEnv)
