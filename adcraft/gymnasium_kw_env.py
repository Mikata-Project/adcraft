"""Gymnasium environment for keyword auctions."""
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np

from adcraft import bidding_simulation as bsim
from adcraft import gymnasium_kw_utils as utils
from adcraft import rust

from adcraft.synthetic_kw_helpers import (
    nonneg_int_normal_sampler,
    probify,
    nonnegify,
)

Obs = Space
Act = Space


class BiddingSimulation(gym.Env):
    """The main Gymnasium class for implementing Reinforcement Learning Agents environments.

    The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the step() and reset() functions. An environment can be partially or fully observed by single agents. For multi-agent environments, see PettingZoo.

    ---
    The main API methods that users of this class need to know are:

    step() - Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.

    reset() - Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info.

    render() - Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text.

    close() - Closes the environment, important when external software is used, i.e. pygame for rendering, databases

    ---
    Environments have additional attributes for users to understand the implementation

    action_space - The Space object corresponding to valid actions, all valid actions should be contained within the space.

    observation_space - The Space object corresponding to valid observations, all valid observations should be contained within the space.

    reward_range - A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode. The default reward range is set to .

    spec - An environment spec that contains the information used to initialize the environment from gymnasium.make()

    metadata - The metadata of the environment, i.e. render modes, render fps

    np_random - The random number generator for the environment. This is automatically assigned during super().reset(seed=seed) and when assessing self.np_random.
    """

    def __init__(
        self,
        keyword_config: Optional[Dict] = None,
        num_keywords: int = 10,
        budget: float = 1000.0,
        render_mode: Optional[str] = None,
        loss_threshold: float = 10000.0,
        max_days: int = 60,
        updater_params: List[List] = [["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        updater_mask: Optional[List[bool]] = None,
        **kwargs,
    ) -> None:
        """
        num_keywords (int): The number of keywords that will be randomly initialized and bid upon.

        budget (float): The budget for each timestep of bidding. Bidding ends in a given timestep if the whole budget has been spent

        render_mode (None or "ansi"): The output from bidding can be rendered as text with "ansi" or not at all with None

        seed (Optional int): Initializes np_random Generator that drives the randomness in the environment

        loss_threshold (float, default = 10000.0): If the agent receives a cumulative loss bigger than loss_threshold, the environment will be truncated. This is essentially a meta-budget over multiple timesteps, though it's not an immediate hard cutoff of bidding. The agent truncates at the end of the timestep they exceed the loss threshold. The timestep this occurs in still runs to completion. So the maximum possible loss of the bidding agent is (loss_threshold + budget).
        """
        super(BiddingSimulation, self).__init__()
        self.keyword_config = keyword_config
        self.num_keywords = num_keywords
        self.budget = budget
        # self.keywords, self.keyword_params = utils.sample_random_keywords(
        #     num_keywords, self.np_random)
        self.action_space = utils.get_action_space(self.num_keywords)
        self.observation_space = utils.get_observation_space(
            self.num_keywords, self.budget
        )
        self.max_days = max_days
        self.loss_threshold = loss_threshold
        self.metadata = {"render_modes": ["ansi"]}

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        ), f'Specified render_mode of ({render_mode}) is not in the allowed options of ({", ".join(self.metadata["render_modes"])})'
        self.render_mode = render_mode

        # reset generates the keywords used in step. and is required after init.
        self._have_keywords = False
        self._current_text = "New start\n"  # used in render method
        self.updater_params = updater_params
        self.updater_mask = updater_mask
        if self.updater_mask is not None:
            self.num_updates = np.sum(self.updater_mask)
        self.init_volumes = None

    def set_updater_mask(self, new_updater_mask: List[bool]) -> None:
        """Replace updater mask with new one and update relevant parameters."""
        assert len(new_updater_mask) == self.num_keywords, (
            f"Updater mask length ({len(new_updater_mask)})\n"
            + "must match number of keywords ({self.num_keywords}) to be applied."
        )
        self.updater_mask = new_updater_mask
        self.num_updates = np.sum(self.updater_mask)

    def update_keywords(self) -> None:
        r"""Update the volume, click rate and conversion rate of every updater_masked keyword.

        Update the mean volume by a uniform additive step bewteen \pm initial_volume * update_params[0][1].
        Assumes volume gets sampled from a nonnegative wrapped normal distribution.
        Volume min clipped to be >= 0.

        Update the ctr by a uniform multiplicative step bewteen 1 \pm update_params[1][1].
        Update the cvr by a uniform multiplicative step bewteen 1 \pm update_params[2][1].
        ctr and cvr are clipped to [0,1] though they can be more than 1 in real life.
        """
        if self.updater_mask is None:
            return
        assert len(self.updater_mask) == self.num_keywords, (
            f"Updater mask length ({len(self.updater_mask)})\n"
            + "must match number of keywords ({self.num_keywords}) to be applied."
        )

        updates = [
            self.np_random.uniform(-v[1], v[1], size=(self.num_updates,))
            for v in self.updater_params
        ]  # all updates sampled uniformly from their parameters. Same update applied to all kws.
        if self.init_volumes is None:
            self.init_volumes = [p[0][1] for p in self.keyword_params]
        for keyword, params, mask, vol_coeff, ctr_coeff, cvr_coeff, init_volume in zip(
            self.keywords,
            self.keyword_params,
            self.updater_mask,
            *updates,
            self.init_volumes,
        ):
            if mask:
                params[0] = (
                    nonnegify(params[0][0] + vol_coeff * init_volume),
                    params[0][1],
                )
                keyword.volume_sampler = nonneg_int_normal_sampler(
                    self.np_random, *params[0]
                )
                keyword.buyside_ctr *= 1 + ctr_coeff
                keyword.buyside_ctr = probify(keyword.buyside_ctr)
                params[3] *= 1 + ctr_coeff
                keyword.sellside_paid_ctr *= 1 + cvr_coeff
                keyword.sellside_paid_ctr = probify(keyword.sellside_paid_ctr)
                params[4] *= probify(1 + cvr_coeff)
                params[3] = probify(params[3])
                params[4] = probify(params[4])

    def step(self, action: Act) -> Tuple[Obs, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (terminated or truncated), it is necessary to call reset() to reset this environment's state for the next episode.

        PARAMETERS:
        action (Act): A Dictionary with two fields.
        "whether_to_bid" (Iterable[bool]) with a length equal to num_keywords.
        ...True -> a keyword bid will be submitted,
        ...False -> no bid will be submitted.
        "keyword_bids" (Iterable[float] >= 0.01) with length equal to num_keywords. Only bids with a corresponding True in "whether_to_bid" will actually be submitted. Each bid submitted will have a minimum value of $0.01 and will be truncated to 2 decimal places.
        "budget" (Optional[float]): Changes budget for this timestep and onward if provided

        RETURNS:
        observation (Obs) - A Dictionary with 7 fields
        "impressions" (np.array) integer number of impressions for each keyword bid upon
        "buyside_clicks" (np.array) integer number of clicks for each keyword bid upon, <= impressions
        "cost"  (np.array) nonnegative float spent each keyword bid upon, sum <= budget
        "sellside_conversions" (np.array) integer number of revenue generating conversions for each keyword bid upon, <= buyside_clicks
        "revenue" (np.array) nonnegative float earned as revenue on each keyword bid upon
        "cumulative_profit" (float) total profit earned from *every* timestep of bidding
        "days_passed" (int) the number of total timesteps that the agent has acted upon keywords

        reward (float) - The net profit for the current timestep. sum of revenues - sum of costs

        terminated (bool) -  Whether the bidder has bid for the final time, i.e. maximum days of bidding has elapsed. If true, the user needs to call reset().

        truncated (bool) - Whether the bidder has cumulatively lost more money than the loss threshold. If true, the user needs to call reset().

        info (dict) - some auxillary information that might be helpful for debugging or logging
        "bids": bids for each keyword bid upon
        "bidding_outcomes": all the computed outcomes from bidding including each individual click cost and individual conversion revenues
        "keyword_params":  the parameters used to initialize each keyword bid upon
        """
        assert (
            self._have_keywords
        ), "reset required, need to generate keywords to bid on"
        budget_array = action.get("budget", self.budget)
        bid_array = action.get("keyword_bids")
        self.budget = np.round(budget_array, 2).astype(float)
        # TODO: I removed this as it was causing training errors.
        # I don't think we need this, but I don't know what I'm
        # breaking by removing it either.
        # self.observation_space = utils.get_observation_space(
        #     self.num_keywords, self.budget)
        keywords = []
        bids = []
        keyword_params = []
        # TODO: Same fix as below for whether_to_bid
        # for bidding, bid, keyword, param in zip(action["whether_to_bid"], action["keyword_bids"], self.keywords, self.keyword_params):
        for bid, keyword, param in zip(bid_array, self.keywords, self.keyword_params):
            if True:
                # TODO: Fix the whether_to_bid action. It is currently ignored to make it work.
                # if bidding:
                keywords.append(keyword)
                bids.append(round(np.maximum(bid, 0.01), 2))
                keyword_params.append(param)

        bidding_outcomes = bsim.simulate_epoch_of_bidding_on_campaign(
            keywords=keywords, bids=bids, budget=self.budget
        )

        profits = rust.sum_list([kw["profit"] for kw in bidding_outcomes])
        self.cumulative_profit += profits
        # lost too much to keep bidding
        truncated = self.cumulative_profit < -self.loss_threshold

        self.current_day += 1
        terminated = self.current_day >= self.max_days

        reward = profits

        observations = dict(
            impressions=np.array([kw["impressions"] for kw in bidding_outcomes]),
            buyside_clicks=np.array([kw["buyside_clicks"] for kw in bidding_outcomes]),
            cost=np.array([rust.sum_list(kw["costs"]) for kw in bidding_outcomes]),
            sellside_conversions=np.array(
                [kw["sellside_conversions"] for kw in bidding_outcomes]
            ),
            revenue=np.array(
                [rust.sum_list(kw["revenues"]) for kw in bidding_outcomes]
            ),
            cumulative_profit=np.array([self.cumulative_profit]),
            days_passed=np.array([self.current_day]),
        )

        self.update_keywords()  # new keyword params
        info = {
            "bids": bids,
            "bidding_outcomes": rust.repr_outcomes_py(bidding_outcomes),
            "keyword_params": utils.repr_all_params(self.keyword_params),
        }

        if self.render_mode == "ansi":
            self._current_text = (
                f"Time step: {self.current_day}/{self.max_days},   "
                + f"Average profit per kw in step: {profits/self.num_keywords:.2f},   "
                + f"Budget: {self.budget}   "
                + f"Total profit in step: {profits:.2f},   "
                + f"Cumulative profit: {self.cumulative_profit:.2f}\n"
            )

        if truncated:
            self._current_text += (
                "Bidding simulation truncated early, we spent too much.\n"
                + f"Our allowed spend was ({self.loss_threshold:.2f}),\n"
                + f"but our cumulative loss was ({self.cumulative_profit:.2f})"
            )

        return observations, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Obs, dict]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the state space and learns a generalised policy about the environment. This randomness can be controlled with the seed parameter otherwise if the environment already has a random number generator and reset() is called with seed=None, the RNG is not reset.

        Therefore, reset() should (in the typical use case) be called with a seed right after initialization and then never again.

        PARAMETERS:
        seed (optional int) - The seed that is used to initialize the environment's PRNG (np_random). If the environment does not already have a PRNG and seed=None (the default option) is passed, a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom). However, if the environment already has a PRNG and seed=None is passed, the PRNG will not be reset. If you pass an integer, the PRNG will be reset even if it already exists. Usually, you want to pass an integer right after the environment has been initialized and then never again. Please refer to the minimal example above to see this paradigm in action.

        options (optional dict) - usable fields to modify class attributes
        "max_days" (int)
        "render_mode" (None or "ansi")
        "loss_threshold" (float)


        RETURNS:
        observation (Obs) - A Dictionary with 7 fields, all of the values will be zero since no bids have been made and no time ahs passed yet.
        "impressions" (np.array) integer number of impressions for each keyword bid upon
        "buyside_clicks" (np.array) integer number of clicks for each keyword bid upon, <= impressions
        "cost"  (np.array) nonnegative float spent each keyword bid upon
        "sellside_conversions" (np.array) integer number of revenue generating conversions for each keyword bid upon, <= buyside_clicks
        "revenue" (np.array) nonnegative float earned as revenue on each keyword bid upon
        "cumulative_profit" (float) total profit earned from *every* timestep of bidding
        "days_passed" (int) the number timesteps that the agent has acted upon keywords

        info (dictionary) - "keyword_params" (str): parameters for initializing current keywords
        """
        super(BiddingSimulation, self).reset(seed=seed)
        # resample keywords if rng changes
        if seed is not None or not self._have_keywords:
            if self.keyword_config is not None:
                (
                    self.keywords,
                    self.keyword_params,
                ) = utils.sample_implicit_keywords_from_quantile_dfs(
                    self.num_keywords, self.np_random, self.keyword_config
                )
            else:
                self.keywords, self.keyword_params = utils.sample_random_keywords(
                    self.num_keywords, self.np_random
                )
            self.keyword_params = [list(kp) for kp in self.keyword_params]
            self._have_keywords = True
        # reset parameters changed with options
        if options:
            self.max_days = options.get("max_days", self.max_days)

            rm = options.get("render_mode", self.render_mode)
            if rm is None or rm in self.metadata["render_modes"]:
                self.render_mode = rm

            self.loss_threshold = options.get("loss_threshold", self.loss_threshold)
        # reset non-optional parameters
        self.current_day = 0
        self.cumulative_profit = 0.0
        self._current_text = "Reset environment\n\nNew start\n"
        observations = dict(
            impressions=np.array([0 for _ in range(self.num_keywords)]),
            buyside_clicks=np.array([0 for _ in range(self.num_keywords)]),
            cost=np.array([0.0 for _ in range(self.num_keywords)]),
            sellside_conversions=np.array([0 for _ in range(self.num_keywords)]),
            revenue=np.array([0.0 for _ in range(self.num_keywords)]),
            cumulative_profit=self.cumulative_profit,
            days_passed=self.current_day,
        )
        # New ##############
        observations = self.observation_space.sample()
        for i, v in observations.items():
            observations[i] = abs(v * 0)
        # End New ########
        info = {"keyword_params": utils.repr_all_params(self.keyword_params)}

        return observations, info

    def render(self) -> Optional[str]:
        """Use this function to get a summary of bidding. Only ansi text is supported for now.

        TODO: add "human" mode which renders graphs that update as bidding occurs
        """
        if self.render_mode == "ansi":
            return self._current_text

    def close(self):
        """Close any open resources that were used by the environment."""
        pass


def bidding_sim_creator(env_config: Dict) -> BiddingSimulation:
    """Unwrap config file into env parameters and construct it."""
    return BiddingSimulation(**env_config)
