"""Helper functions for generating keyword samplers."""
from typing import Callable, Iterable, Tuple, Union

import numpy as np

from adcraft import rust


# helpers
def probify(x: Union[float, np.array]) -> Union[float, np.array]:
    """Force number between 0 and 1."""
    t = type(x)
    out = np.clip(x, 0.0, 1.0).astype(t)
    if isinstance(out, t):
        return out
    return t(out)


def nonnegify(x: Union[int, float, np.array]) -> Union[int, float, np.array]:
    """Force number to be greater than or equal to zero."""
    t = type(x)
    out = np.maximum(x, 0.0).astype(t)
    if isinstance(out, t):
        return out
    return t(out)


# function wrappers
def probified(f):
    """Wrap probify function around function f."""

    def helper(*args):
        return probify(f(*args))

    return helper


def nonnegified(f):
    """Wrap nonnegify function around function f."""

    def helper(*args):
        return nonnegify(f(*args))

    return helper


def intified(f):
    """Wrap int function around function f."""

    def helper(*args):
        return int(f(*args))

    return helper


def generic_cost(x: float, rng: np.random.Generator) -> float:
    """Calculate cost with some added noise."""
    # mean increasing with bid, and goes to 0 as bid goes to 0
    mean_cost = np.sqrt(x) / 4 + x / 2
    # noise increasing with bid, and goes to 0 as bid goes to 0
    cost_noise = rng.normal(0, 1e-10 + np.sqrt(x) / 6)
    # bound cost below by 0 and above by bid
    return np.around(np.clip(mean_cost + cost_noise, 0.0, x), 2).astype(float)


def rev_normal(mean_revenue: float, std_dev: float, rng: np.random.Generator):
    """Sample revenue from a normal distribution."""
    return lambda n: np.around(
        np.maximum(rng.normal(mean_revenue, std_dev, n), 0.01).astype(float), 2
    )


def coinflips(
    heads_prob: float, length: int, rng: np.random.Generator = np.random.default_rng()
) -> np.array:
    """Simulate flipping a coin."""
    return rng.random((length,)) <= heads_prob


# helper to compute beta distribution's beta parameter given a mean m and
# setting alpha = 1.0
def beta_param(m: float):
    """Compute beta distribution's beta parameter."""
    return (1.0 - m) / m


def sigmoid(x, s, t):
    """Calculate sigmoid function given x, s, t."""
    return 1.0 / (1.0 + np.exp(-s * (x - t)))


def bid_abs_normal(
    bid_loc: float, scale: float, rng: np.random.Generator, lowest_bid: float = 0.0
):
    """Return a sampler for absolute value of floats distributed by normal distribution."""
    return lambda s, n: np.around(
        np.maximum(np.abs(rng.normal(bid_loc, scale, (s, n))), lowest_bid).astype(
            float
        ),
        2,
    )


def bid_abs_laplace(
    bid_loc: float, scale: float, rng: np.random.Generator, lowest_bid: float = 0.0
):
    """Return a sampler for absolute value of floats distributed by Laplace distribution."""
    return lambda s, n: np.around(
        np.maximum(np.abs(rng.laplace(bid_loc, scale, (s, n))), lowest_bid).astype(
            float
        ),
        2,
    )


def nth_price_auction(
    bid: float,
    other_bids: np.array,  # num_auctions x num_bidders
    n: int = 2,  # >= 1
    num_winners: int = 2,
) -> Tuple[int, Iterable[int], Iterable[float]]:
    """
    Simulate a nth price auction.

    Simulate a literal nth price auction,
    where the winner pays the price of the bidder (n-1) places below them.
    Returns:
        impressions (int): number of auctions won
        placements (Iterable[int]): what place your bid got in each auction you won.
            zero for you had highest bid, 1 for second highest, etc. down to num_winners
            len(placements) == impressions
        costs (Iterable[float]): the price paid for each auction won.
            (k+n-1)th bid price for kth place

    Arguments:
        bid (float): the bid that a user is willing to pay
        other_bids ([num_auctions x num_bidders] array of floats): These are what "bid"
            will be compared against to determine impressions and costs
        n (int): assumed >= 1, this is the n in nth price auction.
            n=1: 1st price, you pay what you bid if you win.
            n=2: 2nd price, you pay the bid just below yours if you win.
            n=3: 3rd price, you pay the bid two below yours if you win,
            etc.
        num_winners (int): assumed >= 1, the number of different winning placements
            possible. num_winners=2 corresponds to there are two ad spots and first
            place winner gets top spot, 2nd place gets next spot.
    """
    n_bidders = other_bids.shape[1]

    # sort the top bids for each auction
    # so that we can efficiently search for our place and compute cost
    if n_bidders >= num_winners + n:
        top_n_bids = np.sort(
            np.partition(other_bids, -(num_winners + n))[:, -(num_winners + n) :]
        )  # smallest to largest
    else:
        # not enough bidders in auction to calculate all the places,
        # solve this by appending dummy bids of zero.
        # example: 3rd price in an auction with 2 bidders would be 0
        zeros = np.zeros((other_bids.shape[0], num_winners + n - n_bidders))
        top_n_bids = np.sort(np.hstack((zeros, other_bids)))

    impressions = 0
    placements = []
    costs = []
    for auction in top_n_bids:
        index = np.searchsorted(
            auction, bid
        )  # returns index of where bid would be placed in sorted list
        if index > n:
            impressions += 1
            placements.append(num_winners + n - index)
            if n > 1:
                cost_index = int(np.maximum(index - (n - 1), 0))
                costs.append(auction[cost_index])
            else:
                costs.append(bid)
    costs = np.array(costs)
    placements = np.array(placements)
    return impressions, placements, costs


def nonneg_int_normal_sampler(
    rng, the_mean: Union[int, float], std: Union[int, float]
) -> Callable[[], int]:
    """Return a sampler for non-negative integers from a clipped normal distribution."""
    # if the normal draw is negative, output 0 volume, else cast output to int and
    # use that

    def nonnegative_int_normal() -> int:
        return rust.nonneg_int_normal_sampler(the_mean, std)

    return nonnegative_int_normal
