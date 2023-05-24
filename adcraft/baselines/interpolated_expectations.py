"""Baseline methods to compare models against in simulation environment."""
import numpy as np

import torch
from torch import Tensor

from typing import Dict, List, Optional, Tuple


def bidstr(bid: float) -> str:
    """Convert float into a str usable as a dict key."""
    return str(round(float(bid), 2))


def compute_cpc(cost: float, clicks: int) -> float:
    """Compute the average cpc when possible."""
    if clicks > 0:
        return cost / clicks
    return float("nan")


def update_ave_clicks_cache(cache: Dict, bid: float, clicks: int) -> None:
    """Update dict of previous {bid -> (ave_clicks, number_of_observation_generating_average)}.

    Mutates cache["ave_clicks"] to include
    key=bid, value = (new average, 1 + prev number) if a valid observation is made,
    otherwise leaves unchanged
    """
    bid_key = bidstr(bid)
    if cache["ave_clicks"].get(bid_key) is None:
        # first time bidding this value, add cpc and clicks
        cache["ave_clicks"][bid_key] = [clicks, 1]
    else:
        # have bid this before, append new cpc, clicks combo to history
        ave_clicks = cache["ave_clicks"][bid_key][0]
        num_click_obs = cache["ave_clicks"][bid_key][1]
        if not np.isnan(clicks):
            cache["ave_clicks"][bid_key][0] = (clicks + ave_clicks * num_click_obs) / (
                1 + num_click_obs
            )
            cache["ave_clicks"][bid_key][1] = 1 + num_click_obs


def update_ave_cpc_cache(cache: Dict, bid: float, cpc: float, clicks: int) -> None:
    """Update dict of previous {bid -> (ave_cpc, number_of_observation_generating_average)}.

    Mutates cache["ave_cpc"] to include key=bid, value = (new average, 1 + prev number) if a valid observation is made, otherwise leaves unchanged.
    """
    bid_key = bidstr(bid)
    if cache["ave_cpc"].get(bid_key) is None:
        ave_cpc = cpc
        if clicks > 0:
            # first time bidding this value, add cpc and clicks
            cache["ave_cpc"][bid_key] = [ave_cpc, 1]
    else:
        # have bid this before, append new cpc, clicks combo to history
        ave_cpc = cache["ave_cpc"][bid_key][0]
        num_cpc_obs = cache["ave_cpc"][bid_key][1]
        new_cpc = cpc
        if not np.isnan(new_cpc):
            cache["ave_cpc"][bid_key][0] = (new_cpc + ave_cpc * num_cpc_obs) / (
                1 + num_cpc_obs
            )
            cache["ave_cpc"][bid_key][1] = 1 + num_cpc_obs


def process_rpc_and_update_cache(
    ave_rpc, new_obs, num_rpc_obs, cache: Dict
) -> Tuple[float, Dict]:
    """Update the cached value of average revenue per buyside click.

    Returns nan if there have never been any observations of rpc.
    """
    cached_rpc = cache["ave_rpc"]
    num_cached_rpc_obs = cache["num_rpc_obs"]
    if not torch.isnan(ave_rpc) and (num_cached_rpc_obs + num_rpc_obs > 0):
        rpc = (ave_rpc * num_rpc_obs + cached_rpc * num_cached_rpc_obs) / max(
            [1, num_rpc_obs + num_cached_rpc_obs]
        )
        cache["num_rpc_obs"] = new_obs + num_cached_rpc_obs
        cache["ave_rpc"] = rpc
    elif num_cached_rpc_obs + num_rpc_obs > 0:
        rpc = cached_rpc
    else:
        rpc = float("nan")
    return rpc, cache


def process_sctr_and_update_cache(
    ave_sctr, new_obs, num_sctr_obs, cache: Dict
) -> Tuple[float, Dict]:
    """Update the cached value of average paid sellside conversion rate."""
    cached_sctr = cache["ave_sctr"]
    num_cached_sctr_obs = cache["num_sctr_obs"]
    all_obs = num_sctr_obs + num_cached_sctr_obs
    all_convs = ave_sctr * num_sctr_obs + cached_sctr * num_cached_sctr_obs
    if not torch.isnan(ave_sctr) and (num_cached_sctr_obs + num_sctr_obs > 0):
        sctr = (all_convs) / max([1, all_obs])

        cache["num_sctr_obs"] = int(new_obs > 0) + num_cached_sctr_obs
        cache["ave_sctr"] = sctr
    else:
        sctr = cached_sctr
    return sctr, cache


def update_cached_rpc_and_sctr(observations: Tensor, cache: Dict) -> None:
    """Use bids' observations and cache to estimate average rpc and average sctr."""
    # Compute revenue per click ASSUMING independent of bid.
    buyside_clicks = observations[:, :, 1]
    sellside_conversions = observations[:, :, 3]
    revenue = observations[:, :, 4]

    # for only when buyside_clicks > 0
    sellside_conversions[buyside_clicks <= 0] = float("nan")
    buyside_clicks[buyside_clicks <= 0] = float("nan")

    # Added robustness average out an extra buyside click with 50% chance of clicking.
    # In a better model this would give naive confidence bounds instead of taking an average.
    ave_sctrs = torch.reshape(
        torch.nanmean(sellside_conversions / (buyside_clicks), dim=0), (-1,)
    )

    ave_sctrs = torch.maximum(ave_sctrs, Tensor([0.00]))

    revenue[sellside_conversions <= 0] = float("nan")
    revenue[torch.isnan(sellside_conversions)] = float("nan")

    rpcs = revenue / sellside_conversions
    ave_rpscs = torch.reshape(torch.nanmean(rpcs, dim=0), (-1,))

    _, cache = process_rpc_and_update_cache(
        ave_rpscs,
        # 1 if observed rev for most recent bid, 0 else
        int(~torch.isnan(revenue[-1])),
        # number of total observations of revenue
        torch.sum((~torch.isnan(revenue))),
        cache,
    )

    def nan_to_0(f: float) -> int:
        if torch.isnan(f):
            return 0
        else:
            return int(f)

    _, cache = process_sctr_and_update_cache(
        ave_sctrs,
        nan_to_0(observations[-1, :, 1]),
        nan_to_0(torch.nansum(buyside_clicks)),
        cache,
    )


def cache_to_bid_interpolation_points(cache: Dict) -> Tuple[List, List]:
    """Get sorted unique bids and also average outcomes for each bid."""
    unique_bids = []
    ave_values = []
    for bid in np.arange(0.01, 3.01, 0.01):
        if cache.get(bidstr(bid), False):
            # add observed bids to the unique bids, along with mean of that bids outcomes
            # TODO: could use quantiles here to get naive uncertainty given observations
            unique_bids.append(bid)
            ave_values.append(cache[bidstr(bid)][0])
    return unique_bids, ave_values


def get_empirical_average_rev_per_buyside_click() -> Tuple[float, float]:
    """Return hard-coded rev per buyside and sellside values.

    These psuedo-values can be tuned by replacing them with empirical data.
    First value is the revenue per buyside click.
    Second value is the revenue per sellside click.
    """
    return 0.3, 0.7


def get_expected_rev_per_buyside_click(cache: Dict) -> float:
    """Compute revenue per conversion * sellside paid conversion rate if possible.

    Otherwise returns an empirical average from experiment data.
    """
    if cache["num_rpc_obs"] < 1 and cache["num_sctr_obs"] < 1:
        (
            expected_rev_per_buyside_click,
            _,
        ) = get_empirical_average_rev_per_buyside_click()
    elif cache["num_rpc_obs"] < 1:
        (
            _,
            expected_rev_per_sellside_conversion,
        ) = get_empirical_average_rev_per_buyside_click()
        expected_rev_per_buyside_click = float(
            expected_rev_per_sellside_conversion
        ) * float(cache["ave_sctr"])
    else:
        expected_rev_per_buyside_click = float(cache["ave_rpc"]) * float(
            cache["ave_sctr"]
        )
    return float(expected_rev_per_buyside_click)


def smoothed(values: np.array) -> np.array:
    """Smooth values out with a convolution against a triangle/hat function."""
    window_func = np.bartlett(min([5, max([1, len(values) - 1])]))
    mass = np.sum(window_func)
    if mass > 0:
        window_func /= np.sum(window_func)
    else:
        window_func = [1]
    return np.convolve(values, window_func, mode="same")


def full_cache_update(
    bids: Tensor,
    observations: Tensor,
    cache: Dict,
) -> Dict:
    """Update rpc, sctr, ave_cpc, and ave_clicks cache entries with observations."""
    # Compute sctr, and revenue per sclick ASSUMING independent of bid.
    update_cached_rpc_and_sctr(observations, cache)
    # get most recent buyside_clicks and costs
    b_clicks = torch.reshape(observations[:, :, 1], (-1,))
    b_clicks[torch.isnan(b_clicks)] = 0
    costs = torch.reshape(observations[:, :, 2], (-1,))

    most_recent_cpc = compute_cpc(float(costs[-1]), float(b_clicks[-1]))
    most_recent_clicks = b_clicks[-1]
    most_recent_bid = bids[-1]

    # update cached estimates of clikcs per bid and cpc per bid
    update_ave_cpc_cache(cache, most_recent_bid, most_recent_cpc, most_recent_clicks)
    update_ave_clicks_cache(cache, most_recent_bid, most_recent_clicks)

    return cache


def get_expected_profit_per_bid_from_cache(
    cache: Dict,
    allowed_bids: np.array = np.linspace(0.01, 3.0, 300),
) -> Tuple[Tensor, Tensor]:
    """Compute expected cost and profit for all allowed bids."""
    # Compute sctr, and revenue per sclick ASSUMING independent of bid.
    expected_rev_per_buyside_click = get_expected_rev_per_buyside_click(cache)

    # convert dict to sorted list of values amd corresponding bids
    unique_bids_cpc, ave_cpcs = cache_to_bid_interpolation_points(cache["ave_cpc"])
    unique_bids_clicks, ave_clicks = cache_to_bid_interpolation_points(
        cache["ave_clicks"]
    )
    assert np.all(np.diff(unique_bids_cpc) > 0)
    assert np.all(np.diff(unique_bids_clicks) > 0)

    if np.any(unique_bids_cpc):
        # interpolate values over all bids linearly if we have observations
        cpc_per_bid = np.interp(
            allowed_bids,
            unique_bids_cpc,
            smoothed(ave_cpcs),
            left=0.01,
            right=np.max(ave_cpcs),
        )

        clicks_per_bid = np.interp(
            allowed_bids,
            unique_bids_clicks,
            smoothed(ave_clicks),
            left=ave_clicks[0],
            right=ave_clicks[-1],
        )
    else:
        # with no data, cpc is assumed to increase linearly with bid,
        # but be very small so that model is curious enough to try higher bids.
        cpc_per_bid = 0.9 * allowed_bids
        clicks_per_bid = 1.0
    # we add 0.01 to expected clicks to avoid model thinking
    # no observations = 0 since that throws off estimates.
    expected_margins = (-cpc_per_bid + expected_rev_per_buyside_click) * (
        0.01 + clicks_per_bid
    )
    expected_costs = cpc_per_bid * (0.01 + clicks_per_bid)

    return expected_margins, expected_costs


def get_empty_cache() -> Dict:
    """Returns an empty cache for average values."""
    return {
        "ave_rpc": 0.0,
        "num_rpc_obs": 0,
        "ave_sctr": 0.4,
        "num_sctr_obs": 0.0,
        "ave_cpc": {},
        "ave_clicks": {},
    }


class NaiveInterpolationStrategy:
    """Estimates revenue per buyside click, clicks per bid, and cost per bid to find profit.

    The agent then uses that information to sample bids believed to be profitable.
    revenue per buyside click is (rpc * sctr) where
    rpc is calculated as average revenue per paid sellside conversion and
    sctr is calculated as average paid conversions per buyside click

    clicks per bid and cost per bid are both averaged individually for each unique past bid.
    those averages are interpolated to estimate the average clicks and average cpc for all
    allowed bids.

    Profit for each bid is then estimated by
    (0.01 + clicks_per_bid) * (- cpc_per_bid + expected_rev_per_buyside_click)

    Bids with expected profit above a given threshold will be sampled proportional to how high above the threshold they are. If no bids are expected to be above the threshold, then the agent will bid 0.01.
    """

    def __init__(
        self,
        num_keywords: int,
        profit_acquisition_threshold: float = -0.2,
        allowed_bids: np.array = np.linspace(0.01, 3.00, 300),
        initial_caches: Optional[List[Dict]] = None,
        seed: Optional[int] = None,
        bid_step: float = 0.03,
    ) -> None:
        """Initialize the agent on a particular number of keywords.

        Set the threshold for profit above which the agent bids.
        Optionally fill the cache with preseeded values and set the agent's rng seed.
        """
        if initial_caches is None:
            self.caches = [get_empty_cache() for _ in range(num_keywords)]
        else:
            self.caches = initial_caches
        self.observation_keys = [
            "impressions",
            "buyside_clicks",
            "cost",
            "sellside_conversions",
            "revenue",
        ]
        self.profit_acquisition_threshold = profit_acquisition_threshold
        self.allowed_bids = allowed_bids
        self.bid_step = bid_step
        self.profit_beliefs = None
        self.cost_beliefs = None
        self.acquisition_function = None
        self.rng = np.random.default_rng(seed)

    def update_single_cache(
        self, kw_index: int, prev_bid: float, prev_observation: Dict
    ) -> None:
        """Update the cache for a given keyword using bid observation pair."""
        observation = [prev_observation[k][kw_index] for k in self.observation_keys]
        # profit = revenue - cost
        observation.append(observation[4] - observation[2])
        data_y = torch.reshape(Tensor(observation), (1, 1, -1))
        data_x = torch.unsqueeze(Tensor([prev_bid]), dim=1)
        self.caches[kw_index] = full_cache_update(data_x, data_y, self.caches[kw_index])

    def get_expected_margin_from_cache(
        self, kw_index: int
    ) -> Tuple[np.array, np.array]:
        """Compute the profit per bid for each keyword from cached information."""
        (expected_margins, expected_costs) = get_expected_profit_per_bid_from_cache(
            self.caches[kw_index], self.allowed_bids
        )

        return expected_margins, expected_costs

    def get_profit_acquisition_function(
        self, expected_margin: np.array, index: int
    ) -> Optional[np.array]:
        """Return optional probability distribution to sample from.

        Likelihood is proportional to bidwise profitability above the threshold.
        """
        threshold = -(
            1
            / (
                1
                + self.caches[index]["num_rpc_obs"]
                + self.caches[index]["num_sctr_obs"] / 5
            )
        ) * np.abs(self.profit_acquisition_threshold)
        acquisition_function = np.maximum(expected_margin, threshold) - threshold
        observed_bids = [float(b) for b in self.caches[index]["ave_clicks"].keys()]
        observed_bids.append(0.03)
        max_observed_bid = max(observed_bids)
        mob_index = int(100 * (max_observed_bid + self.bid_step) - 1)
        L = len(acquisition_function)
        end_index = min([L, mob_index])
        acquisition_function[end_index:] = 0.0
        mass = np.sum(acquisition_function[:end_index])
        if mass > 0:
            self.acquisition_function = acquisition_function / mass
            return acquisition_function / mass
        # returns None when all bids expected to be below acquisition threshold for profit
        # i.e. when keyword is believed to be unprofitable

    def update_all_caches(self, prev_action: None, prev_observations: Dict) -> None:
        """Update the cache for each keyword."""
        for i, prev_bid in enumerate(prev_action["keyword_bids"]):
            self.update_single_cache(i, prev_bid, prev_observations)

    def sample_action(self) -> Dict:
        """Sample a bid from those that appear profitable above threshold."""
        bids = []
        expected_cost = 0.0
        expected_profit = 0.0

        for i, _ in enumerate(self.caches):
            margins, costs = self.get_expected_margin_from_cache(i)
            acquisition_function = self.get_profit_acquisition_function(
                margins, index=i
            )
            if acquisition_function is None:
                # expected to be always too negative profit
                bids.append(0.01)
            else:
                index = self.rng.choice(
                    list(range(len(self.allowed_bids))), p=acquisition_function
                )
                bids.append(self.allowed_bids[index])
                if self.caches[i]["num_sctr_obs"] > 0:
                    expected_cost += costs[index]
                else:
                    expected_cost += self.allowed_bids[index]
                if self.caches[i]["num_rpc_obs"] > 0:
                    expected_profit += margins[index]
        self.profit_beliefs = expected_profit
        self.cost_beliefs = expected_cost
        assert self.cost_beliefs is not None, f"debug {expected_cost}"
        if expected_profit > 0:
            budget = 1.5 * max([min([expected_cost, 10000]), 1000])
        elif expected_profit > len(self.caches) * self.profit_acquisition_threshold:
            budget = max([min([expected_cost, 10000]), 1000])
        else:
            budget = 1000
        return {"budget": budget, "keyword_bids": np.array(bids)}


class NaiveZeroMarginStrategy:
    """Estimates revenue per buyside click.

    The agent then uses that information to sample bids believed to be profitable.
    revenue per buyside click is (rpc * sctr) where
    rpc is calculated as average revenue per paid sellside conversion and
    sctr is calculated as average paid conversions per buyside click

    Profit is assumed due to cost per click being less than or equal to bid.
    So expected profit is the difference between the bid and what we pay in the second price auction.

    This would be optimal in a non-repeated second price auction with no fancy bells and whistles from the auctioneer.

    In reality, we have both fancy bells and whistles as well as repeating, since every search is a new auction.
    """

    def __init__(
        self,
        num_keywords: int,
        default_expected_revenue_per_conversion: float = 3.0,
        initial_caches: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the agent on a particular number of keywords.

        Optionally fill the cache with preseeded values and set the agent's rng seed.
        """
        if initial_caches is None:
            self.caches = [get_empty_cache() for _ in range(num_keywords)]
        else:
            self.caches = initial_caches
        self.observation_keys = [
            "impressions",
            "buyside_clicks",
            "cost",
            "sellside_conversions",
            "revenue",
        ]
        self.rng = np.random.default_rng(seed)
        self.max_bids = np.full((num_keywords,), 0.01)
        self.prev_bids = None
        self.default_rpc = default_expected_revenue_per_conversion

    def update_all_caches(self, prev_action, prev_observation):
        """Update the values for rpc and sctr in the caches."""
        self.prev_bids = prev_action["keyword_bids"]
        for i, bid in enumerate(prev_action["keyword_bids"]):
            observation = [prev_observation[k][i] for k in self.observation_keys]
            # profit = revenue - cost
            observation.append(observation[4] - observation[2])
            data_y = torch.reshape(Tensor(observation), (1, 1, -1))

            update_cached_rpc_and_sctr(data_y, self.caches[i])

    def sample_action(self):
        """If no rpc are observed, ramp up the bid, otherwise bid the rpc."""
        bids = np.zeros((len(self.max_bids),))
        budget = 0.0
        for i, old_bid in enumerate(self.prev_bids):
            if self.caches[i]["num_rpc_obs"] < 1:
                if self.rng.random() <= 1 / np.sqrt(self.caches[i]["num_sctr_obs"]):
                    # just step up previous bid with prob proportional to sqrt of observations
                    new_bid = max([0.01, min([self.max_bids[i] + 0.03, 3.0])])
                    self.max_bids[i] = new_bid
                    budget += 1
                else:
                    new_bid = self.caches[i]["ave_sctr"] * self.default_rpc
                    budget += 2
            else:
                expected_rpc = get_expected_rev_per_buyside_click(self.caches[i])
                new_bid = expected_rpc
                budget += 3
            bids[i] = new_bid
        return {"budget": 100 * budget, "keyword_bids": bids}
