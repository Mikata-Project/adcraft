"""Compute NCP and AKNCP."""
from typing import Tuple

import numpy as np

from adcraft.gymnasium_kw_utils import ExplicitKeywordGeneratingParams
from adcraft.synthetic_kw_classes import ExplicitKeyword, ImplicitKeyword


def get_explicit_kw_bid_cpc_impressions(
    explicit_keyword: ExplicitKeyword, bid_array: np.array, n_samples: int = 2048
):
    """Compute the auction win rate and approximate cost for each bid in bid array."""
    impression_rate = np.array([explicit_keyword.impression_rate(b) for b in bid_array])
    N = len(bid_array)
    med_cost_per_bid = np.array([np.median(explicit_keyword.cost_per_buyside_click(bid, n_samples)) for bid in bid_array])
    return impression_rate, med_cost_per_bid


def get_implicit_kw_bid_cpc_impressions(
    implicit_keyword: ImplicitKeyword, bid_array: np.array, n_samples: int = 2048
) -> Tuple[np.array, np.array]:
    """Compute the approximate auction win rate and cost for each bid in bid array.

    Compute the expected second price for that bid conditioned on having won.
    The conditional second price is the average of sampled bids below a given bid.
    """
    second_prices = np.reshape(np.sort(implicit_keyword.sample_bids(n_samples)), (-1,))
    # print(second_prices)
    indices = np.searchsorted(second_prices, bid_array, side="right")

    impression_rates = indices / n_samples
    indices = np.minimum(indices, n_samples - 1)
    mean_prices = np.cumsum(second_prices) / np.arange(1, n_samples + 1, 1)
    expected_cpc_per_bid = mean_prices[indices]

    return impression_rates, expected_cpc_per_bid


def get_max_expected_bid_profits(
    kw_params: ExplicitKeywordGeneratingParams,
    expected_cpc_per_bid: np.array,
    expected_impression_rate_per_bid: np.array,
) -> Tuple[float, float]:
    """Return maximum expected profit over all bids and proportion of bids with positive EV.

    This isn't the maximum possible profit. More can be earned due to stochasticity/variance.
    kw_params: ((vol_mean, vol_std), 50%_impression_bid, 50%_impression_slope, bctr, sctr, mean_revenue, std_revenue)
    expected_profits = vol_mean * impression_rate * bctr * (sctr * mean_revenue - costs).
    """
    expected_profits = np.maximum(
        kw_params[0][0]
        * expected_impression_rate_per_bid
        * kw_params[3]
        * (kw_params[4] * kw_params[5] - expected_cpc_per_bid),
        0.0,
    )
    return (
        max([0.0, expected_profits.max()]),
        np.sum(expected_profits > 0) / len(expected_cpc_per_bid), np.argmax(expected_profits)
    )


def compute_AKNCP(kw_profits: np.array, ideal_profits: np.array) -> float:
    """Return the median ratio of keyword profit to maximum expected profit for that keyword.

    This isn't the maximum possible profit. More can be earned due to stochasticity/variance.
    kw_params: ((vol_mean, vol_std), 50%_impression_bid, 50%_impression_slope, bctr, sctr, mean_revenue, std_revenue)
    expected_profits = vol_mean * impression_rate * bctr * (sctr * mean_revenue - costs).
    """
    denominator = ideal_profits.copy()
    denominator[denominator <= 0] = 1.0
    denominator = denominator.mean(axis=0)
    # if ideal_profits is same as keyword_profits, then that should be 1.
    return np.median(kw_profits.mean(axis=0) / denominator)


def compute_NCP(kw_profits: np.array, ideal_profits: np.array) -> float:
    """Return the ratio of actual profit to ideal profits."""
    denominator = ideal_profits.sum()
    if denominator <= 0.0:
        denominator = 1.0
    return kw_profits.sum() / denominator
