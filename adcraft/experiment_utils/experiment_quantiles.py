"""Generate and load experiment quantiles."""
from typing import Dict, List

import pandas as pd


def singleton_mmm_dict(param: str, value: List) -> Dict:
    """Return dict of columns-value pairs corresponding to param in singleton quantile dataframe."""
    return {
        f"count_{param}": [3],
        f"min_{param}": [value[0]],
        f"median_{param}": [value[1]],
        f"max_{param}": [value[2]],
    }

def get_generic_sparsity_experiment_dict() -> Dict:
    return {
        "vol": [64, 128, 256],
        "ave_cpc": [0.3, 0.55, 1],
        "std_cpc": [0.01, 0.15, 0.3],
        "bctr": [0.1, 0.5, 0.9],
        "sctr": [0.1, 0.5, 0.9],
        "rpsc": [0.3, 1.0, 1.5],
        "std_rpsc": [0.01, 0.15, 0.3],
    }

def dict_to_singleton_quantile_triple_dict(data_dict: Dict) -> Dict:
    singleton_quantile_dict = {}
    for k, v in data_dict.items():
        singleton_quantile_dict.update(singleton_mmm_dict(k, v))
    experiment_quantiles = pd.DataFrame(singleton_quantile_dict)
    return experiment_quantiles

def generate_simple_experiment_quantiles(mean_volume: int, cvr: float) -> pd.DataFrame:
    """Generates a csv with a single quantile bin's min, median, and max for each param.

    Mean volume and paid conversion rate are user-specified, and shared across all kws.
    """
    data_dict = get_generic_sparsity_experiment_dict()
    data_dict["vol"] = [mean_volume, mean_volume, mean_volume]
    data_dict["sctr"] = [cvr, cvr, cvr]

    return dict_to_singleton_quantile_triple_dict(data_dict)

def generate_simple_bctr_experiment_quantiles(ctr: float, cvr: float) -> pd.DataFrame:
    """Generates a csv with a single quantile bin's min, median, and max for each param.

    buyside click-through rate and paid conversion rate are user-specified, and shared across all kws.
    """
    data_dict = get_generic_sparsity_experiment_dict()
    data_dict["bctr"] = [ctr, ctr, ctr]
    data_dict["sctr"] = [cvr, cvr, cvr]

    return dict_to_singleton_quantile_triple_dict(data_dict)

def generate_simple_vol_bctr_experiment_quantiles(mean_volume: int, ctr: float) -> pd.DataFrame:
    """Generates a csv with a single quantile bin's min, median, and max for each param.

    Mean volume and buyside click-through rate are user-specified, and shared across all kws.
    """
    data_dict = get_generic_sparsity_experiment_dict()
    data_dict["vol"] = [mean_volume, mean_volume, mean_volume]
    data_dict["bctr"] = [ctr, ctr, ctr]

    return dict_to_singleton_quantile_triple_dict(data_dict)


def make_experiment_quantiles(keyword_config: Dict) -> None:
    """Wrap generate_simple_experiment_quantiles to use as a make_quantile_func."""
    v = keyword_config["mean_volume"]
    cvr = keyword_config["conversion_rate"]
    outer_dir = keyword_config["outer_directory"]
    generate_simple_experiment_quantiles(v, cvr).to_csv(f"{outer_dir}/{v}_{cvr}.csv")


def load_experiment_quantiles(keyword_config: Dict) -> pd.DataFrame:
    """Loads in the singleton quantile bin created by make_experiment_quantiles.

    Supply as load_quant_func arg of keyword_config for the experiments.
    """
    v = keyword_config["mean_volume"]
    cvr = keyword_config["conversion_rate"]
    outer_dir = keyword_config["outer_directory"]
    return pd.read_csv(f"{outer_dir}/{v}_{cvr}.csv")


def make_bctr_experiment_quantiles(keyword_config: Dict) -> None:
    """Wrap generate_simple_experiment_quantiles to use as a make_quantile_func."""
    ctr = keyword_config["clickthrough_rate"]
    cvr = keyword_config["conversion_rate"]
    outer_dir = keyword_config["outer_directory"]
    generate_simple_bctr_experiment_quantiles(ctr, cvr).to_csv(f"{outer_dir}/{ctr}_{cvr}.csv")


def load_bctr_experiment_quantiles(keyword_config: Dict) -> pd.DataFrame:
    """Loads in the singleton quantile bin created by make_experiment_quantiles.

    Supply as load_quant_func arg of keyword_config for the experiments.
    """
    ctr = keyword_config["clickthrough_rate"]
    cvr = keyword_config["conversion_rate"]
    outer_dir = keyword_config["outer_directory"]
    return pd.read_csv(f"{outer_dir}/{ctr}_{cvr}.csv")

def make_vol_bctr_experiment_quantiles(keyword_config: Dict) -> None:
    """Wrap generate_simple_experiment_quantiles to use as a make_quantile_func."""
    ctr = keyword_config["clickthrough_rate"]
    vol = keyword_config["mean_volume"]
    outer_dir = keyword_config["outer_directory"]
    generate_simple_vol_bctr_experiment_quantiles(vol, ctr).to_csv(f"{outer_dir}/{vol}_{ctr}.csv")


def load_vol_bctr_experiment_quantiles(keyword_config: Dict) -> pd.DataFrame:
    """Loads in the singleton quantile bin created by make_experiment_quantiles.

    Supply as load_quant_func arg of keyword_config for the experiments.
    """
    ctr = keyword_config["clickthrough_rate"]
    vol = keyword_config["mean_volume"]
    outer_dir = keyword_config["outer_directory"]
    return pd.read_csv(f"{outer_dir}/{vol}_{ctr}.csv")
