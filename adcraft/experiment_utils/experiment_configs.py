import os
from pathlib import Path
import adcraft.gymnasium_kw_env as kw_sim
from adcraft.experiment_utils.experiment_quantiles import (
    make_experiment_quantiles, load_experiment_quantiles)


### Here we set the experiment mode   
experiment_mode = 'production_training'


NUM_KEYWORDS = 100
MAX_DAYS = 60


dense_env_config = dict(
    keyword_config={
        "outer_directory": Path.cwd().as_posix(),
        "mean_volume": 128,
        "conversion_rate": 0.8,
        "make_quant_func": make_experiment_quantiles,
        "load_quant_func": load_experiment_quantiles
        },
        num_keywords=NUM_KEYWORDS,
        max_days=MAX_DAYS,
        updater_params=[["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        updater_mask=None
)

semi_dense_env_config = dict(
    keyword_config={
        "outer_directory": Path.cwd().as_posix(),
        "mean_volume": 64,
        "conversion_rate": 0.8,
        "make_quant_func": make_experiment_quantiles,
        "load_quant_func": load_experiment_quantiles
        },
        num_keywords=NUM_KEYWORDS,
        max_days=MAX_DAYS,
        updater_params=[["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        updater_mask=None
)

sparse_env_config = dict(
    keyword_config={
        "outer_directory": Path.cwd().as_posix(),
        "mean_volume": 64,
        "conversion_rate": 0.1,
        "make_quant_func": make_experiment_quantiles,
        "load_quant_func": load_experiment_quantiles
        },
        num_keywords=NUM_KEYWORDS,
        max_days=MAX_DAYS,
        updater_params=[["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        updater_mask=None
)

very_sparse_env_config = dict(
    keyword_config={
        "outer_directory": Path.cwd().as_posix(),
        "mean_volume": 16,
        "conversion_rate": 0.1,
        "make_quant_func": make_experiment_quantiles,
        "load_quant_func": load_experiment_quantiles
        },
        num_keywords=NUM_KEYWORDS,
        max_days=MAX_DAYS,
        updater_params=[["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        updater_mask=None
)


non_stationary_dense_env_config = dict(
    keyword_config={
        "outer_directory": Path.cwd().as_posix(),
        "mean_volume": 128,
        "conversion_rate": 0.8,
        "make_quant_func": make_experiment_quantiles,
        "load_quant_func": load_experiment_quantiles
        },
        num_keywords=NUM_KEYWORDS,
        max_days=MAX_DAYS,
        updater_params=[["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        updater_mask=[True]*NUM_KEYWORDS
)

non_stationary_sparse_env_config = dict(
    keyword_config={
        "outer_directory": Path.cwd().as_posix(),
        "mean_volume": 64,
        "conversion_rate": 0.1,
        "make_quant_func": make_experiment_quantiles,
        "load_quant_func": load_experiment_quantiles
        },
        num_keywords=NUM_KEYWORDS,
        max_days=MAX_DAYS,
        updater_params=[["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        updater_mask=[True] * NUM_KEYWORDS
)