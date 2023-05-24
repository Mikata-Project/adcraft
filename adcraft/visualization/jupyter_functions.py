"""Simple visualization functions for bidding in a jupyter environment."""
from typing import List, Optional

from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt


def show_keyword_profits(
    kw_profits: List[np.array],
    bids: List[np.array],
    absolute_max_bid: Optional[float] = None,
    replace_output: bool = True,
) -> None:
    """Replace or add to jupyter output with three rows of figures.

    Each row displays an identical image of the bids placed for each keyword.
    Darkest navy blue is a bid of 0 and brightest orange is absolute_max_bid.

    The top right image is the mean profits per step over
    ..(top) negative profit keywords,
    ..(middle) positive profit keywords,
    ..(middle) all keywords. +
    The colors are
    ..Red = negative,
    ..Green = positive, and
    ..White = 0.
    ..with normalization such that plus or minus the
    .. max magnitude of the first two of those determine the darkest green/red.
    In a 'good' run:
    ..the (top) should be roughly reddest at the left and fade to white over time.
    ..the (middle) should over time get greener until it has a roughly consistent green level
    ..the (bottom) might fluctuate between green and red at first before getting
    ..greener on average until it hopefully remains dark green.
    ..If the max profitability is less than first few losses observed, then
    ..it might never get that dark even when doing very well.

    On the middle row, left image shows normalized profit for each keyword.

    On the bottom row, left image shows sign of profit for each keyword.
    """
    im_profits = np.array(kw_profits)
    sign_profits = np.sign(im_profits)

    aspect = max([1 / 4, min([len(bids) / len(bids[0]), 4])])  # between 1/4 and 4
    H = max([3, min([6, len(bids[0]) / 10])])
    fig, axs = plt.subplots(
        3,
        2,
        sharex=True,
        sharey=True,
        figsize=(H * 2 * aspect, 3 * H),
    )
    if absolute_max_bid is None:
        absolute_max = np.array(bids).max()
    else:
        absolute_max = absolute_max_bid

    axs[0][0].imshow(
        np.array(bids).transpose(), interpolation=None, vmin=0, vmax=absolute_max
    )
    profs = np.mean(im_profits.transpose(), axis=0)
    neg_profs = [
        np.mean(im_profits[i, im_profits[i, :] < 0], axis=0) for i in range(len(bids))
    ]
    neg_profs = np.array([0 if np.isnan(p) else p for p in neg_profs])
    pos_profs = [
        np.mean(im_profits[i, im_profits[i, :] > 0], axis=0) for i in range(len(bids))
    ]
    pos_profs = np.array([0 if np.isnan(p) else p for p in pos_profs])
    np_rows = [neg_profs] * int(np.floor(len(bids[0]) / 3))
    pp_rows = [pos_profs] * int(np.floor(len(bids[0]) / 3))
    prof_rows = [profs * len(bids[0])] * int(np.ceil(len(bids[0]) / 3))

    pmax = max(
        [np.abs(profs).max(), np.abs(pos_profs[:]).max(), np.abs(neg_profs[:]).max()]
    )
    axs[0][1].imshow(
        np.vstack(tuple(np_rows + pp_rows + prof_rows)),
        cmap="PiYG",
        interpolation=None,
        vmin=-pmax - 0.001,
        vmax=pmax + 0.001,
    )

    axs[1][0].imshow(
        im_profits.transpose(),
        cmap="PiYG",
        interpolation=None,
        vmin=-np.abs(im_profits).max(),
        vmax=np.abs(im_profits).max(),
    )

    axs[1][1].imshow(
        np.array(bids).transpose(), interpolation=None, vmin=0, vmax=absolute_max
    )

    axs[2][0].imshow(
        sign_profits.transpose(),
        cmap="PiYG",
        interpolation=None,
        vmin=-np.abs(sign_profits).max(),
        vmax=np.abs(sign_profits).max(),
    )
    axs[2][1].imshow(
        np.array(bids).transpose(), interpolation=None, vmin=0, vmax=absolute_max
    )
    fig.tight_layout()

    if replace_output:
        clear_output(wait=True)
    plt.show()


def print_agg_metric(metric, name: str = "profit") -> None:
    """Print summary statistics of measurements of a given metric."""
    print(f"total {name}: {np.sum(metric)}")
    print(f"max {name} per timestep: {np.max(metric)}")
    print(f"min {name} per timestep: {np.min(metric)}")
    print(f"mean {name} per time step {np.mean(metric)}")
    print(f"std dev {name} per time step {np.std(metric)}")


def show_cumulative_rewards(rewards) -> None:
    """Plot the cumulative sum of values.

    Print the summary statistics for individual values.
    """
    plt.figure(figsize=(12, 5))

    print_agg_metric(rewards, name="rewards")
    plt.subplot(111)
    plt.plot(np.cumsum(rewards))
    plt.title("cumulative_rewards")
    plt.grid(visible=True, which="both", axis="both")
    plt.show()
