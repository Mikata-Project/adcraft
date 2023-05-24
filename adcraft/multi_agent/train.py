"""Functions for training in Multi-Agent setting."""
import logging

from ray.tune.registry import register_env

from adcraft.gymnasium_kw_env import BiddingSimulation
from adcraft.multi_agent.env import make_multi_flat
from adcraft.wrappers.flat_array import FlatArrayWrapper


def basic_policy_mapping_fn(agent_id: int, episode, worker, **kwargs):
    """Convert agent_id to str and return."""
    return str(agent_id)


def multi_train(config_list: list, policy_list: list, epochs: int = 1) -> dict:
    """
    Train multiple agents for evaluation.

    Results can be found here:
        result["sampler_results"]["policy_reward_mean"]

    Usage:
    import os

    from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
    from ray.rllib.algorithms.sac import SACConfig, SACTorchPolicy

    from adcraft.multi_agent.train import multi_train, basic_policy_mapping_fn

    ppo_config = (
        PPOConfig()
        .framework("torch")
        .training(
            model={"vf_share_layers": True},
            vf_loss_coeff=0.01,
            num_sgd_iter=6,
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    sac_config = (
        SACConfig()
        .framework("torch")
        .rollouts(create_env_on_local_worker=True)
        .training(model={"fcnet_hiddens" : [32,32]})  # shape of policy observation to action
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .evaluation(
            evaluation_interval = 1
        )
    )

    config_list = [ppo_config, sac_config]
    policy_list = [PPOTorchPolicy, SACTorchPolicy]

    result = multi_train(config_list, policy_list)
    print(result["sampler_results"]["policy_reward_mean"])

    > {'0': -1.8499999999999994, '1': 15.739999999999998}
    """
    single_dummy_env = BiddingSimulation(config={})
    single_dummy_env = FlatArrayWrapper(single_dummy_env)
    obs_space = single_dummy_env.observation_space
    act_space = single_dummy_env.action_space

    len_config = len(config_list)
    len_policy = len(policy_list)
    assert len_config == len_policy

    env_name = "multi_agent_auction"
    register_env(env_name, lambda _: make_multi_flat(len_config))

    policies = {}
    for i, p in enumerate(policy_list):
        policies[f"{i}"] = (p, obs_space, act_space, {})

    for i, c in enumerate(config_list):
        config_list[i] = (
            c.multi_agent(
                policies=policies,
                policy_mapping_fn=basic_policy_mapping_fn,
                policies_to_train=[f"{i}"],
            ).environment(env_name)
        ).build()

    for i in range(epochs):
        logging.info("== Iteration", i, "==")
        for j in range(len_config):
            logging.info(f"-- Config: {j} --")
            result = config_list[j].train()

    logging.info(f"Final rewards: {result['sampler_results']['policy_reward_mean']}")

    return result
