import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import torch
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.a2c import A2C, A2CConfig
from ray.rllib.algorithms.td3 import TD3, TD3Config
from pathlib import Path


from adcraft.experiment_utils import experiment_configs
import adcraft.gymnasium_kw_env as kw_sim
from adcraft.experiment_utils.experiment_quantiles import (
    make_experiment_quantiles, load_experiment_quantiles)
from adcraft.experiment_utils.experiment_metrics import (
    get_implicit_kw_bid_cpc_impressions, get_max_expected_bid_profits, compute_AKNCP, compute_NCP)
from adcraft.wrappers.flat_array import FlatArrayWrapper

from adcraft.experiment_utils.experiment_metrics import (get_implicit_kw_bid_cpc_impressions, get_max_expected_bid_profits,compute_AKNCP,compute_NCP)

from adcraft.experiment_utils.experiment_configs import (dense_env_config, semi_dense_env_config, very_sparse_env_config, sparse_env_config, non_stationary_sparse_env_config, non_stationary_dense_env_config)


from adcraft.experiment_utils.experiment_configs import NUM_KEYWORDS
from adcraft.experiment_utils.experiment_configs import MAX_DAYS
from adcraft.experiment_utils.experiment_configs import experiment_mode



NUM_KEYWORDS = NUM_KEYWORDS
MAX_DAYS = MAX_DAYS

experiment_type = experiment_mode


if experiment_type == "semi_dense":
    env_config = semi_dense_env_config
elif experiment_type == "sparse":
    env_config = sparse_env_config
elif experiment_type == "very_sparse":
    env_config = very_sparse_env_config
elif experiment_type == "non_stationary_dense":
    env_config = non_stationary_dense_env_config
elif experiment_type == "non_stationary_sparse":
    env_config = non_stationary_sparse_env_config
else: 
    env_config = dense_env_config


from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

############################# RL Agent Configs  #############################

sem_ppo_config = (
    PPOConfig()
    .framework("torch")
    .rollouts(create_env_on_local_worker=True, rollout_fragment_length= 'auto')
    .rollouts(num_rollout_workers=1, num_envs_per_worker = 46)
    .training(gamma = 0.995, lambda_ = 0.95, lr = 0.0001, kl_coeff=1.0, 
    clip_param = 0.5, sgd_minibatch_size= 64, train_batch_size = 2048, 
    num_sgd_iter=20,
    model = {
        "fcnet_hiddens": [32, 32],  # deep and dense network
        "fcnet_activation": "relu",  # activation function
    })
    .environment(env = "FlatArrayAuction", env_config=env_config)
    .evaluation(evaluation_config = {'explore': False}, evaluation_interval = 5, evaluation_duration_unit = 'episodes', evaluation_duration = 5, evaluation_sample_timeout_s = 90.0, evaluation_num_workers =2)
    .resources(num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)


sem_a2c_config = (
    A2CConfig()
    .framework("torch")
    .training(gamma = 0.99, lambda_ = 0.99, lr = 0.001 , microbatch_size = 32, 
    train_batch_size = 2048, grad_clip = 1.0,
    model = {
        "fcnet_hiddens": [256, 256],  # deep and dense network
        "fcnet_activation": "relu",  # activation function
    },
    vf_loss_coeff = 0.5, entropy_coeff = 0.01)
    .rollouts(create_env_on_local_worker=True, rollout_fragment_length= 'auto')
    .rollouts(num_rollout_workers=23, num_envs_per_worker = 2)
    .environment(env = "FlatArrayAuction", env_config=env_config)
    .evaluation(evaluation_config = {'explore': False}, evaluation_interval = 5, evaluation_duration_unit = 'episodes', evaluation_duration = 5, evaluation_sample_timeout_s = 90.0, evaluation_num_workers = 2)
    .resources(num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)


sem_td3_config = (
    TD3Config()
    .framework("torch")
    .training(gamma = 0.995, lr =0.001, train_batch_size = 2048, tau = 0.005, 
    num_steps_sampled_before_learning_starts = 10000, 
    model = {
        "fcnet_hiddens": [400, 300],  # deep and dense network
        "fcnet_activation": "relu",  # activation function
    },
    replay_buffer_config = {
            "type": "MultagentsiAgentReplayBuffer",
            "capacity": 1000000,
            "worker_side_prioritization": False,
        }) 
    .rollouts(create_env_on_local_worker=True, rollout_fragment_length= 'auto')
    .rollouts(num_rollout_workers= 23, num_envs_per_worker = 2)
    .environment(env = "FlatArrayAuction", env_config=env_config)
    .exploration(exploration_config={
            # TD3 uses simple Gaussian noise on top of deterministic NN-output
            # actions (after a possible pure random phase of n timesteps).
            "type": "GaussianNoise",
            # For how many timesteps should we return completely random
            # actions, before we start adding (scaled) noise?
            "random_timesteps": 10000,
            # Gaussian stddev of action noise for exploration.
            "stddev": 0.1,
            # Scaling settings by which the Gaussian noise is scaled before
            # being added to the actions. NOTE: The scale timesteps start only
            # after(!) any random steps have been finished.
            # By default, do not anneal over time (fixed 1.0).
            "initial_scale": 1.0,
            "final_scale": 1.0,
            "scale_timesteps": 1,
        })
    .evaluation(evaluation_config = {'explore': False}, evaluation_interval = 5, evaluation_duration_unit = 'episodes', evaluation_duration = 5, evaluation_sample_timeout_s = 90.0, evaluation_num_workers = 2)
    .resources(num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)

