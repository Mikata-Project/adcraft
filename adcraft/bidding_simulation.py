"""High level functions to simulate a bidding enviornment."""
from typing import Iterable, Optional, TypedDict

import numpy as np

from adcraft.synthetic_kw_classes import Keyword
from adcraft import rust


class BiddingOutcomes(TypedDict):
    """Outcomes returned from simulating a day of bidding.

    Attributes:
        bid (float): used to compute below outcomes
        impressions (int): number of impressions seen at bid value
        impression_share (float): proportion of total possible impressions observed
        buyside_clicks (int):  number of buyside impressions that were clicked on
        costs (Iterable[float]): individual costs for each buyside click
        sellside_conversions (int): number of sellside impressions that resulted in
            revenue
        revenues (Iterable[float]): amount of revenue received for each
            sellside_paid_click
        revenues_per_cost (Iterable[float]): amount of revenue received for each
            buyside_click (same as revenues, but with zeros for buyside clicks
            resulting in no revenue)
        profit (float): total profit observed for the bid price, equal to
            sum(revenues) - sum(costs)
    """

    bid: float
    impressions: int
    impression_share: float
    buyside_clicks: int
    costs: Iterable[float]
    sellside_conversions: int
    revenues: Iterable[float]
    revenues_per_cost: Iterable[float]
    profit: float


# cost and revenue outcomes for fixed bidding value
# TODO: if bidding strategy is mixed instead of pure, need to modify this
# TODO: use placement to give different buyside ctr
def simulate_epoch_of_bidding(
    keyword: Keyword,
    bid: float,
    budget: float = float("inf"),
    n_auctions: Optional[int] = None,
) -> BiddingOutcomes:
    """
    Compute the outcomes of bidding at a fixed price for a given day.

    Compute the outcomes of bidding at a fixed price for a given day, or for a specified
    number of auctions.

    Returns:
        (BiddingOutcomes): the impressions, clicks, cpcs, rpc for buyside clicks and rpc
        for sellside clicks along with total profit.

    Arguments:
        keyword (Implicit or Explicit Keyword): keyword which is bid on.
        bid (float): how much we bid in all the auctions.
        budget (float): how much can be spent before no more impressions are given.
            currently no pacing is simulated, so it's same number of auctions until
            budget is fully spent. by default there's no budget constraint.
        num_auctions (Optional[int]): If None, then number of auctions is sampled from
            the volume distribution. Otherwise a specified int will run that number of
            auctions instead.
    """
    if n_auctions is not None:
        volume = n_auctions
    else:
        volume = int(keyword.sample_volume()[0])

    outcomes = BiddingOutcomes(
        bid=bid,
        impressions=0,
        impression_share=0.0,
        buyside_clicks=0,
        costs=[],
        sellside_conversions=0,
        revenues=[],
        revenues_per_cost=[],
        profit=0.0,
    )
    outcomes["impressions"], placements, click_costs = keyword.auction(
        bid, num_auctions=volume
    )
    if volume > 0:
        outcomes["impression_share"] = outcomes["impressions"] / volume
    else:
        outcomes["impression_share"] = 0.0  # or 1.0?
    # don't use placements here, but could.
    impression_clicks: Iterable[bool] = keyword.sample_buyside_click(
        len(click_costs)
    )  # True for clicked, False for not
    for clicked, cost in zip(impression_clicks, click_costs):
        if clicked:
            if budget >= cost:
                outcomes["buyside_clicks"] += 1
                outcomes["costs"].append(cost)
                budget -= cost
            else:
                break
    # sample the revenues from paid clicks on the sellside
    paid_sellside_clicks: Iterable[bool] = keyword.sample_sellside_paid_click(
        outcomes["buyside_clicks"]
    )
    outcomes["sellside_conversions"] = rust.sum_array_bool(paid_sellside_clicks)
    # list of individual payouts given that a paid sellside click happened
    outcomes["revenues"] = keyword.sample_reward(outcomes["sellside_conversions"])
    # list of ALL payouts given that a buyside click happened (no sellside payout = 0
    # reward in below)
    outcomes["revenues_per_cost"] = rust.list_to_zeros(outcomes["costs"])
    outcomes["revenues_per_cost"][paid_sellside_clicks] = np.array(outcomes["revenues"])
    # total profit observed over a day
    outcomes["profit"] = rust.sum_array(outcomes["revenues"]) - rust.sum_list(
        outcomes["costs"]
    )
    return outcomes


# TODO: fix the type hints below, or change the addable/catable thing
def combine_outcomes(*outcomes: Iterable[BiddingOutcomes]) -> BiddingOutcomes:
    """Combine outcomes together into one result."""
    result = outcomes[0]
    addable_fields = ["impressions", "buyside_clicks", "sellside_conversions", "profit"]
    catable_fields = ["costs", "revenues", "revenues_per_cost"]
    for outcome in outcomes[1:]:
        if result["impressions"] < 1:
            old_vol = 0.0
        else:
            old_vol = np.round(result["impressions"] / result["impression_share"])
        if outcome["impressions"] < 1:
            next_vol = 0.0
        else:
            next_vol = np.round(outcome["impressions"] / outcome["impression_share"])
        for field in addable_fields:
            result[field] += outcome[field]
        for field in catable_fields:
            result[field] = np.concatenate((result[field], outcome[field]))
        vol = old_vol + next_vol
        if vol > 0:
            result["impression_share"] = result["impressions"] / (vol)
        else:
            result["impression_share"] = 0.0
    return result


# TODO: Write better docstring.
def uniform_get_auctions_per_timestep(
    timesteps: int, *kws: Keyword
) -> Iterable[Iterable[int]]:
    """Select auction for each timestep."""
    volumes = []
    volume_step = []
    # print(f"debug kws: {kws}")
    for kw in kws:
        # print(f"debug kw: {kw}")
        volumes.append(int(kw.sample_volume()[0]))
        volume_step.append(volumes[-1] // timesteps)
    auctions_per_timestep = [
        [vol - (timesteps - 1) * v for vol, v in zip(volumes, volume_step)]
    ]
    for t in range(timesteps - 1):
        auctions_per_timestep.append(volume_step)
    return auctions_per_timestep


def simulate_epoch_of_bidding_on_campaign(
    keywords: Iterable[Keyword],
    bids: Iterable[float],
    budget: float = float("inf"),
    auctions_per_timestep: Optional[Iterable[Iterable[int]]] = None,
) -> BiddingOutcomes:
    """
    Compute the outcomes of bidding at a fixed price.

    Compute the outcomes of bidding at a fixed price for a given epoch, or for a
    specified number of auctions.

    Returns:
        (BiddingOutcomes): the impressions, clicks, cpcs, rpc for buyside clicks and rpc
        for sellside clicks along with total profit.

    Arguments:
        keywords (Iterable of Implicit or Explicit Keyword): ordered of the keywords in
            the campaign
        bids (Iterable[float]): how much we bid on each keyword in the same order and
            length as keywords.
        budget (float): how much can be spent before no more impressions are given.
            currently no pacing is simulated, so it's same number of auctions until
            budget is fully spent. By default there's no budget constraint.
        auctions_per_timestep (Iterable[Iterable[int]]): If None, then number of
            auctions is sampled from the volume distribution / the number of t.
    """
    # initialize data
    outcomes_per_keyword = [
        BiddingOutcomes(
            bid=bids[i],
            impressions=0,
            impression_share=0.0,
            buyside_clicks=0,
            costs=[],
            sellside_conversions=0,
            revenues=[],
            revenues_per_cost=[],
            profit=0.0,
        )
        for i, keyword in enumerate(keywords)
    ]
    if auctions_per_timestep is None:
        auctions_per_timestep = uniform_get_auctions_per_timestep(24, *keywords)
    remaining_budget = budget
    # run auctions
    for timestep, auctions in enumerate(auctions_per_timestep):
        for kw_index, keyword in enumerate(keywords):
            new_outcomes = simulate_epoch_of_bidding(
                keyword=keyword,
                bid=bids[kw_index],
                budget=remaining_budget,
                n_auctions=auctions[kw_index],
            )

            remaining_budget -= rust.sum_list(new_outcomes["costs"])

            outcomes_per_keyword[kw_index] = combine_outcomes(
                outcomes_per_keyword[kw_index], new_outcomes
            )
            if remaining_budget <= 0:
                break
        if remaining_budget <= 0:
            break
    return outcomes_per_keyword
