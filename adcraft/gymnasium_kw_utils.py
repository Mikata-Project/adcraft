"""Utility functions for the Gymnasium environment for keywords."""
from typing import Iterable, List, Optional, Tuple

from gymnasium.spaces import Box, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from adcraft import rust
from adcraft.synthetic_kw_classes import Keyword, ExplicitKeyword, ImplicitKeyword
from adcraft.synthetic_kw_helpers import (
    nonneg_int_normal_sampler,
    rev_normal,
    bid_abs_laplace,
)
from adcraft.pull_quantiles_data.quantiles_to_keywords import sample_from_quantiles


ExplicitKeywordGeneratingParams = Tuple[
    Tuple[int, float], float, float, float, float, float, float
]
# ((vol_mean, vol_std), 50%_impression_bid, 50%_impression_slope, bctr, sctr, mean_revenue, std_revenue)

ImplicitKeywordGeneratingParams = Tuple[
    Tuple[int, float], float, float, float, float, float, float
]
# ((vol_mean, vol_std), cost_loc, cost_scale, bctr, sctr, mean_revenue, std_dev_revenue)


def get_action_space(num_keywords: int) -> Dict:
    """Create the environment's action space."""
    # TODO: This works by "trusting" that the Dict is ordered. Refactor later.
    return Dict(
        {
            # "whether_to_bid": MultiBinary(num_keywords),
            "keyword_bids": Box(
                low=0.01, high=float("Inf"), shape=(num_keywords,), dtype=np.float32
            ),
            "budget": Box(low=0.01, high=float("Inf"), shape=(1,), dtype=np.float32),
        }
    )


def get_observation_space(num_keywords, budget) -> Dict:
    """Create the environment's observation space."""
    NonnegativeIntBox = Box(low=0, high=float("Inf"), shape=(num_keywords,), dtype=int)
    CostSpace = Box(low=0, high=budget, shape=(num_keywords,), dtype=np.float32)
    NonnegativeFloatBox = Box(
        low=0, high=float("Inf"), shape=(num_keywords,), dtype=np.float32
    )
    return Dict(
        {
            "impressions": NonnegativeIntBox,
            "buyside_clicks": NonnegativeIntBox,
            "cost": CostSpace,
            "sellside_conversions": NonnegativeIntBox,
            "revenue": NonnegativeFloatBox,
            "cumulative_profit": Box(
                low=-float("Inf"), high=float("Inf"), shape=(1,), dtype=np.float32
            ),
            "days_passed": Box(low=0, high=float("Inf"), shape=(1,), dtype=np.float32),
        }
    )


def generate_keyword_from_params(
    vol: Tuple[int, float],
    imp_intercept: float,
    imp_slope: float,
    bctr: float,
    sctr: float,
    mean_revenue: float,
    std_revenue: float,
    rng: np.random.Generator,
) -> Tuple[ExplicitKeyword, ExplicitKeywordGeneratingParams]:
    """Create keywords with various attribute values."""
    return ExplicitKeyword(
        {
            "rng": rng,
            "impression_thresh": 0.05,
            "impression_bid_intercept": imp_intercept,
            "impression_slope": imp_slope,
            "sellside_paid_ctr": sctr,
            "buyside_ctr": bctr,
            "volume_sampler": nonneg_int_normal_sampler(
                rng, the_mean=vol[0], std=vol[1]
            ),
            # TODO: Pass in seed.
            "cost_per_buyside_click": rust.cost_create,
            "reward_distribution_sampler": rev_normal(
                mean_revenue, std_dev=std_revenue, rng=rng
            ),
        },
        verbose=True,
    ), (vol, imp_intercept, imp_slope, bctr, sctr, mean_revenue, std_revenue)


def get_keywords_from_params(
    pre_params_list: Iterable[ExplicitKeywordGeneratingParams], rng: np.random.Generator
) -> Tuple[List[Keyword], List[ExplicitKeywordGeneratingParams]]:
    """Create kewyords for use in the environment."""
    keywords, params_list = [], []
    for pre_params in pre_params_list:
        keyword, params = generate_keyword_from_params(*pre_params, rng)

        params_list.append(params)
        keywords.append(keyword)

    return keywords, params_list


def sample_random_keywords(
    num_keywords: int, rng: np.random.Generator
) -> Tuple[List[Keyword], List[ExplicitKeywordGeneratingParams]]:
    """Sample num_keywords many ExplicitKeyword's parameters from independent distributions.

    The below lists are sampled using rng provided to initialize a list of keywords. The
    parameters used to generate each keyword are given as output along with the
    generated keywords.
    These distributions are pretty arbitrary. You can replace them with whatever.

    PARAMETERS
    num_keywords (int) - this many keywords will be sampled
    rng (np.random.Generator) - this is the PRNG used to sample the keywords, AND is
        also the rng that is passed into the keywords to use for their own internal
        randomness.
    """
    v_mean_list = (2 ** rng.beta(2, 5, size=num_keywords) * 15 - 1).astype(
        int
    )  # bounded above by 16k, mode = 32
    # below: +1 so zero vol can have rare impressions
    v_std_list = rng.random(size=num_keywords) * 0.5 * (v_mean_list + 1)
    vol_list = [(vm, vs) for vm, vs in zip(v_mean_list, v_std_list)]
    sctr_list = rng.beta(5, 2, size=num_keywords)
    imp_intercept_list = rng.random(size=num_keywords) * 1.5
    mean_revenue_list = rng.beta(2, 5, size=num_keywords) * 1.5
    std_revenue_list = rng.beta(2, 5, size=num_keywords) * mean_revenue_list
    bctr_list = rng.beta(2, 5, size=num_keywords)
    imp_slope_list = rng.beta(5, 5, size=num_keywords) * 25

    return get_keywords_from_params(
        [
            (vol, imp_intercept, imp_slope, bctr, sctr, mean_revenue, std_revenue)
            for vol, imp_intercept, imp_slope, bctr, sctr, mean_revenue, std_revenue in zip(
                vol_list,
                imp_intercept_list,
                imp_slope_list,
                bctr_list,
                sctr_list,
                mean_revenue_list,
                std_revenue_list,
            )
        ],
        rng,
    )


def single_competitor() -> int:
    """Return 1.

    Used for creating implicit keywords with only one competetor.
    Bid distribution then models highest competitor bid
    instead of the distribution of each competitor bids.
    """
    return int(1)


def generate_implicit_keyword_from_params(
    vol: Tuple[int, float],
    cost_loc: float,
    cost_scale: float,
    bctr: float,
    sctr: float,
    mean_revenue: float,
    std_revenue: float,
    rng: np.random.Generator,
) -> Tuple[ImplicitKeyword, ExplicitKeywordGeneratingParams]:
    """Return an ImplicitKeyword Initialized with a set of given parameters."""
    return ImplicitKeyword(
        {
            "rng": rng,
            "bidder_distribution": single_competitor,
            "bid_distribution": bid_abs_laplace(cost_loc, cost_scale, rng),
            "sellside_paid_ctr": sctr,
            "buyside_ctr": bctr,
            "volume_sampler": nonneg_int_normal_sampler(
                rng, the_mean=vol[0], std=vol[1]
            ),
            "reward_distribution_sampler": rev_normal(
                mean_revenue, std_dev=std_revenue, rng=rng
            ),
        },
        verbose=True,
    ), (vol, cost_loc, 1 / cost_scale, bctr, sctr, mean_revenue, std_revenue)


def get_implicit_keywords_from_params(
    pre_params_list: Iterable[ImplicitKeywordGeneratingParams], rng: np.random.Generator
) -> Tuple[List[Keyword], List[ExplicitKeywordGeneratingParams]]:
    """For each parameter in the given list, generate an ImplictKeyword and return them all."""
    keywords, params_list = [], []
    for imp_params in pre_params_list:
        keyword, ex_params = generate_implicit_keyword_from_params(*imp_params, rng)

        params_list.append(ex_params)
        keywords.append(keyword)

    return keywords, params_list


def make_quantile_df_csvs(keyword_config: Dict) -> None:
    """Pull quantile data from data source and store quantiles in csv files locally."""
    outer_directory = keyword_config.get(
        "outer_directory", Path.cwd().as_posix() + "/quantile_dfs/"
    )
    # If they already exist, just use them as is.
    replace = keyword_config.get("replace", False)
    quantiles_folder = keyword_config.get("quantiles_folder", "")
    quant_dir = Path(outer_directory + quantiles_folder)
    if not quant_dir.exists():
        quant_dir.mkdir(parents=True)

    auction_fname = outer_directory + quantiles_folder + "per_auction.csv"
    if replace or not Path(auction_fname).exists():
        # data = quantiles_load()
        # data.to_csv(auction_fname)
        # del data
        raise NotImplementedError(
            "You must implement your own version of make_quantile_df_csvs"
        )
    else:
        print(
            f"Auction quantiles file already exists at {auction_fname}. Replace is False"
        )


def load_quantile_dfs_from_csv(
    keyword_config: Dict,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Return quantile dfs for volume and auction outcomes.

    Read quantile data to data frames from (vol.csv, per_auction.csv)
    which are assumed to be in that folder.
    Returns each DataFrame or None if each files does or doesn't exist respectively.
    """
    outer_directory = keyword_config.get(
        "outer_directory", Path.cwd().as_posix() + "/quantile_dfs/"
    )
    quantiles_folder = keyword_config.get("quantiles_folder")
    auction_file = outer_directory + quantiles_folder + "auction_data.csv"

    if Path(auction_file).exists() and Path(auction_file).is_file():
        auction_data = pd.read_csv(auction_file)
    else:
        auction_data = None
    return auction_data


def sample_implicit_keywords_from_quantile_dfs(
    num_keywords: int, rng: np.random.Generator, keyword_config: Dict
) -> Tuple[List[Keyword], List[ExplicitKeywordGeneratingParams]]:
    """Sample num_keywords many ImplicitKeyword from quantiles.

    Quantiles are pulled from prod data according to keyword_config.

    The below lists are sampled using rng provided to initialize a list of keywords.
    The parameters used to generate each keyword are given as output along with
    the generated keywords.

    PARAMETERS
    num_keywords (int) - this many keywords will be sampled
    rng (np.random.Generator) - this is the PRNG used to sample the keywords, AND is
    also the rng that is passed into the keywords to use for their own internal randomness.
    keyword_config (Dict) - parameters used to determine quantile file location and
    determine how to pull them if relevant data hasn't been pulled yet.
    """
    # params_list will be equivalent to
    # [vol_list, cost_loc_list, cost_scale_list, bctr_list, sctr_list, revenue_list, revenue_std_list]

    load_quantile_from_csv = keyword_config.get(
        "load_quant_func", load_quantile_dfs_from_csv
    )
    make_quantile_csvs = keyword_config.get("make_quant_func", make_quantile_df_csvs)
    if keyword_config.get("quantiles_folder", False):
        data = load_quantile_from_csv(keyword_config)
    else:
        make_quantile_csvs(keyword_config)
        data = load_quantile_from_csv(keyword_config)

    assert (
        data is not None
    ), "Invalid quantile parameters specified in keyword_config for data"

    no_volume_prob = keyword_config.get("no_vol_prob", 0.0)
    params_lists = [
        [
            (int(v), int(1 + rng.random() * 0.5 * v))
            if rng.random() > no_volume_prob and not np.isnan(v)
            else (0, rng.random() * 0.5)
            for v in sample_from_quantiles(
                num_keywords,
                len(data),
                data["min_vol"],
                data["median_vol"],
                data["max_vol"],
                rng,
            )
        ]
    ]

    for param in [
        "ave_cpc",
        "std_cpc",
        "bctr",
        "sctr",
        "rpsc",
        "std_rpsc",
    ]:
        data_param = data[data[f"count_{param}"] > 0].loc[
            :, [f"min_{param}", f"median_{param}", f"max_{param}"]
        ]
        params_lists.append(
            sample_from_quantiles(
                num_keywords,
                len(data_param),
                data_param[f"min_{param}"],
                data_param[f"median_{param}"],
                data_param[f"max_{param}"],
                rng,
            )
        )
        # Assumes standard deviations have been normalized to multipliers on average value.
        # Un-normalize them below.
        if param[:4] == "std_":
            for i in range(num_keywords):
                params_lists[-1][i] = max(
                    [0.01, params_lists[-1][i] * params_lists[-2][i]]
                )
    del data
    return get_implicit_keywords_from_params(
        [
            (vol, cost_loc, cost_scale, bctr, sctr, rev_loc, rev_scale)
            for vol, cost_loc, cost_scale, bctr, sctr, rev_loc, rev_scale in zip(
                *params_lists
            )
        ],
        rng,
    )


def repr_params(params: List[str]) -> str:
    """List out the name and value of each of the parameters passed in to define a keyword."""
    return ",   ".join(
        [
            name + f": {value}"
            for name, value in zip(
                [
                    "volume",
                    "imp_intercept",
                    "imp_slope",
                    "bctr",
                    "sctr",
                    "mean revenue",
                    "std revenue",
                ],
                params,
            )
        ]
    )


def repr_all_params(params_list: List[List[str]]) -> str:
    """List out the name and values for the parameters for a list of keywords' parameters."""
    return "\n".join(
        [
            f"kw{n} params:\n {repr_params(params)}"
            for n, params in enumerate(params_list)
        ]
    )


def flatten_dict_array(obs: "dict[np.ndarray]") -> np.ndarray:
    """Flatten dict of arrays into single numpy array."""
    res_list = []
    for k in sorted(obs.keys()):
        res_list.append(obs[k].ravel())
    flat_obs = np.hstack(res_list)

    return flat_obs


# TODO: Refactor as this does multiple things.
def plot_explicit_kw_properties(keywords, params) -> Tuple[List[float], List[float]]:
    """Samples multiple outcomes for each bid 0.01 to 2.00 and plots the outcomes for each keyword.

    Compute average profit, cost, and revenue for each bid for each keyword to make the plots;
    with extra time proportional to (# keywords * # bids) compute bid optimizing average profit.

    RETURNS
    optimal_bids (List[float]): The bid maximizing the average profit for each keyword.

    optimal_ave_profits (List[float]): The maximum of the average profits for each keyword.
    """
    optimal_bids = []
    optimal_ave_profits = []
    for kw, params in zip(keywords, params):
        print(repr_params(params))
        bid_cents = np.linspace(0.01, 2, 200)
        plt.figure()
        # cost
        ave_cost = [
            params[0][0]
            * kw.impression_rate(x)  # ave volume
            * kw.buyside_ctr
            * np.mean([kw.cost_per_buyside_click(x) for _ in range(100)])
            for x in bid_cents
        ]
        # revenue
        ave_rev = [
            params[0][0]
            * kw.impression_rate(x)  # ave volume
            * kw.buyside_ctr
            * kw.sellside_paid_ctr
            * np.mean(kw.reward_distribution_sampler(50))
            for x in bid_cents
        ]
        ave_profit = [r - c for r, c in zip(ave_rev, ave_cost)]

        # save actions of a static oracle
        opt_ind = np.argmax(ave_profit)
        max_p = ave_profit[opt_ind]
        if max_p >= 0:
            optimal_ave_profits.append(max_p)
            optimal_bids.append(bid_cents[opt_ind])
        else:
            optimal_ave_profits.append(0.0)
            optimal_bids.append(0.0)

        # plot the average cost, revenue, profit, and impression rate
        plt.plot(bid_cents, ave_cost, "r", label="avg cost")
        plt.plot(bid_cents, ave_rev, "g", label="avg revenue")
        plt.plot(bid_cents, ave_profit, "o", label="avg profit")
        plt.plot(
            bid_cents,
            [kw.impression_rate(x) for x in bid_cents],
            "b",
            label="impression share",
        )
        plt.title("average metrics against bid price")
        plt.legend()
        plt.show()

        plt.figure()
        nonneg_profit = [r - c if r - c > 0 else 0.0 for r, c in zip(ave_rev, ave_cost)]
        nonneg_roi = [
            (r - c) / c if r - c > 0 and c > 0 else 0.0
            for r, c in zip(ave_rev, ave_cost)
        ]
        m = sum(nonneg_profit)
        if m > 0:
            # plots a normalized distribution of profitable bids
            ideal_profit_distr = np.array([p / m for p in nonneg_profit])
            plt.plot(bid_cents, ideal_profit_distr, "o", label="profit utility")
        else:
            plt.plot(
                [bid_cents[0], bid_cents[-1]], [0.0, 0.0], "r", label="no avg profit"
            )
        m = sum(nonneg_roi)
        if m > 0:
            # plots a normalized distribution of return for bids with positive return
            ideal_roas_distr = np.array([p / m for p in nonneg_roi])
            plt.plot(
                bid_cents, ideal_roas_distr, "o", label="return on investment utility"
            )
        plt.title("normalized distribution of average *positive* utility against bid")
        plt.legend()
        plt.show()
    plt.show()
    return optimal_bids, optimal_ave_profits
