"""Tests for gymnasium enviornment."""
import gymnasium as gym
import pytest

from adcraft.gymnasium_kw_env import BiddingSimulation


# TODO: Replace some of below with ray's check_env function to make more robust.
@pytest.mark.unit
def test_type() -> None:
    env = BiddingSimulation()
    if not isinstance(env, gym.Env):
        raise ValueError(
            f"Env must be of type gymnasium.Env, but instead is of type {type(env)}."
        )


@pytest.mark.unit
def test_observation() -> None:
    env = BiddingSimulation()
    if not hasattr(env, "observation_space"):
        raise AttributeError("Env must have an observation_space attribute!")


@pytest.mark.unit
def test_observation_space() -> None:
    env = BiddingSimulation()
    if not isinstance(env.observation_space, gym.spaces.Space):
        raise ValueError("Observation space must be a gymnasium.Space!")


@pytest.mark.unit
def test_action() -> None:
    env = BiddingSimulation()
    if not hasattr(env, "action_space"):
        raise AttributeError("Env must have action_space attribute!")


@pytest.mark.unit
def test_action_space() -> None:
    env = BiddingSimulation()
    if not isinstance(env.action_space, gym.spaces.Space):
        raise ValueError("Action space must be a gymnasium.Space!")


@pytest.mark.unit
@pytest.mark.parametrize("s", [None, 1])
def test_reset(s: int) -> None:
    env = BiddingSimulation()
    env.reset(seed=s)


@pytest.mark.unit
def test_reset_obs() -> None:
    env = BiddingSimulation()
    reset_obs, reset_infos = env.reset()
    assert env.observation_space.contains(reset_obs)


@pytest.mark.unit
def test_step_obs() -> None:
    env = BiddingSimulation()
    reset_obs, reset_infos = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    # TODO: Below is hacky version of ray's convert_element_to_space_type function.
    for k, v in reset_obs.items():
        next_obs[k] = next_obs[k].astype(v.dtype)
    assert env.observation_space.contains(next_obs)
