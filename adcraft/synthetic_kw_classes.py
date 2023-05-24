"""Functions for generating traffic units at the keyword level."""
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypedDict,
)

import numpy as np

from adcraft.synthetic_kw_helpers import (
    beta_param,
    coinflips,
    intified,
    nonnegified,
    nonnegify,
    nth_price_auction,
    probify,
)
from adcraft import rust


# @dataclass
class KeywordParams(TypedDict, total=False):
    """Parameters used to initialize the properties of a Keyword.

    If any of these are missing, a default will be used.
    Generally the default will be sampled from some distribution on initialization of a Keyword.
    For some of these, there are [alternative parameters] to modify how the default will be sampled.
    Fields denoted as [alternative parameters] are ignored when the actual parameter is provided.

    Attributes:
        rng (np.random.Generator): a numpy random Generator used for
            samples of this keyword's distributions

        volume_sampler (Callable[[]->int]): distribution whose samples give number of auctions on
            this keyword for whatever time interval we care about, e.g. a day
            [alternative parameter]
            volume (int): will make the volume sampler a constant function that always returns the
                          volume provided

        buyside_ctr (0 <= float <= 1): chance of buyside click given that there is an impression
            [alternative parameter]
            buyside_ctr_avg (0 <= float <= 1): the expected value of a beta distribution that a random buyside_ctr will be sampled from if not explicit

        sellside_paid_ctr (0 <= float <= 1): chance of sellside revenue given a buyside click
            [alternative parameter]
            sellside_paid_ctr_avg (0 <= float <= 1): the expected value of a beta distribution that a random buyside_ctr will be sampled from if not explicit

        reward_distribution_sampler (Callable[[int], Iterable[float]]):
            given some number of times revenue was earned on the sellside, returns that many rewards

        # used in ExplicitKeyword only
        impression_rate (Callable[[float], float]): takes in a bid and returns impression chance
            [alternative parameters]
            impression_thresh (0 <= float < 0.5):
                postprocessing for impression rates generated from other parameters
                impression_rate below the threshold will be replaced with 0, and above (1-impression_thresh) will be sent to 1.
            impression_bid_intercept (float):
                This value is the bid at which we achieve 50% impression rate
            impression_slope (float): This is the tangent to the sigmoid at the
                impression_bid_threshold above

        cost_per_buyside_click (Callable[[float], float]):
            given a bid returns a sampled cost of buyside click. If not provided a default model proportional to sqrt(bid)+gaussian noise

        # used in ImplicitKeyword only
        bidder_distribution (Callable[[], int]): a distribution for the number of bidders
            participating in auctions on the keyword.
            [alternative parameters]
            # Default samples from a binomial distribution.
            max_bidders (int): maximum possible bidders in this keyword's auctions.
            participation_rate (float): probability each bidder participates

        bid_distribution (Callable[[int, int], np.array]):
            the distribution that each bidder's bids are sampled from.  Currently they're all identically distributed.
            [alterantive parameters]
            # Default samples from a laplace distribution, then take absolute value.
            bid_loc (float): center/peak of laplace distribution used
            bid_scale (0 <= float): how concentrated/spread out the distribution is at the peak.
    """

    rng: np.random.Generator
    seed: int

    volume_sampler: Callable[[], int]
    volume: int

    buyside_ctr: float
    buyside_ctr_avg: float

    sellside_paid_ctr: float
    sellside_paid_ctr_avg: float

    reward_distribution_sampler: Callable[[int], Iterable[float]]
    reward_cdf_pts: List[List[float]]

    # ExplicitKeyword parameters
    impression_rate: Callable[[float], float]
    impression_thresh: float
    impression_bid_intercept: float
    impression_slope: float

    cost_per_buyside_click: Callable[[float], float]

    # ImplicitKeyword parameters
    bidder_distribution: Callable[[], int]
    max_bidders: int
    participation_rate: float

    bid_distribution: Callable[[int, int], np.array]
    bid_loc: float
    bid_scale: float


class Keyword:
    """
    A class to represent a keyword for bidding.

    Meant to be subclassed by either explicit or implicit implementations of cost/impressions.

    ...
    Attributes:
        rng (np.random.Generator): a numpy random Generator used for
            samples of this keyword's distributions, default initialized with seed
        volume_sampler (Callable[[]->int]): distribution whose samples give number of auctions on
            this keyword for whatever time interval we care about, e.g. a day
        buyside_ctr (0 <= float <= 1): chance of buyside click given that there is an impression
        sellside_paid_ctr (0 <= float <= 1): chance of sellside revenue given a buyside click
        reward_distribution_sampler (Callable[[int], Iterable[float]]):
            given some number of times revenue was earned on the sellside, returns that many rewards


    Methods
    -------
    sample_volume(self, n=1)->np.array:
        returns n samples from the volume distribution of the Keyword

    auction(self, bid: float, num_auctions: int) -> Tuple[int, Iterable[int], Iterable[float]]:
        samples the number of impressions, placements, and individual buyside costs per click for num_auction many auctions

    sample_buyside_click(self, n=1)->np.array:
        samples a length n boolean array. True for each buyside click, and False for non-click.
        each element corresponds to single impression

    sample_sellside_paid_click(self, n=1)->np.array:
        samples a length n boolean array. True if sellside impression resulted in revenue-earning
        sellside click(s), and False otherwise.

    sample_reward(self, n=1)->np.array:
        samples length n array of sellside revenues given that revenue was earned.
    """

    def __init__(self, params: KeywordParams = {}, verbose=False) -> None:
        """Establish sampling functions and other RNG related things."""
        Keyword._validate_parameters(KeywordParams(params), verbose)

        self.rng = Keyword._rng_init(params)

        self.volume_sampler: Callable[[], int] = Keyword._volume_sampler_init(params)
        self.buyside_ctr: float = Keyword._buyside_ctr_init(self.rng, params)
        self.sellside_paid_ctr: float = Keyword._sellside_paid_ctr_init(
            self.rng, params
        )
        self.reward_distribution_sampler: Callable[
            [int], Iterable[float]
        ] = Keyword._reward_distribution_sampler_init(self.rng, params)

    def sample_volume(self, n: int = 1) -> np.array:
        """
        Sample volume n number of times.

        Returns:
            an int array of samples from the volume distribution for a keyword

        Keyword arguments:
            n (int): the number of samples desired in the array (default: 1)
        """
        return np.array([self.volume_sampler() for _ in range(n)])

    def auction(
        self, bid: float, num_auctions: int
    ) -> Tuple[int, Iterable[int], Iterable[float]]:
        """
        Simulate results of num_auctions many auctions.

        Returns:
            impressions (int): number of impressions = number of auctions won
            placements (Iterable[int]): location of each auction won
            costs (Iterable[float]): price of an adclick on each impression

        Arguments:
            bid (float): how much we bid in all the auctions
            num_auctions (int): the number of auctions run = max number of impressions
        """
        # this must be overwritten by subclass
        raise (
            NotImplementedError(
                "Must use Keyword subclass with auction implementation such as ExplicitKeyword or ImplicitKeyword"
            )
        )

    def sample_buyside_click(self, n: int = 1) -> np.array:
        """
        Sample buyside clicks n number of times.

        Returns:
            a boolean array of buyside clicks for each of n impressions.
            values are True for impression led to buyside click
            and False if there was no click given the impression

        Keyword arguments:
            n (int): the length of the output array, and also the number of impressions
        """
        return coinflips(self.buyside_ctr, n, self.rng)

    def sample_sellside_paid_click(self, n: int = 1) -> np.array:
        """
        Sample sellside paid clicks n times.

        Returns:
            a boolean array of sellside paid clicks for each of n buyside clicks.
            values are True for buyside click led to sellside paid click
            and False if there was no sellside click given the buyside click

        Keyword arguments:
            n (int): the length of the output array, and also the number of buyside clicks
        """
        return coinflips(self.sellside_paid_ctr, n, self.rng)

    def sample_reward(self, n: int = 1) -> np.array:
        """
        Sample reward n times.

        Returns:
            float array of n rewards sampled from the keyword's reward distribution

        Keyword arguments:
            n (int): the length of the output array, and also the number of sellside paid clicks
        """
        return np.array(self.reward_distribution_sampler(n)).reshape((n,))

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve __dict__ class attribute.

        Returns:
            a dictionary of all the exposed attributes, which also works as a param dict for
            instantiating an identical keyword.
        """
        return self.__dict__

    @staticmethod
    def _validate_seed(params: KeywordParams, verbose: bool = False) -> None:
        """Validate seed is appropriate."""
        seed = params.get("seed")
        if seed is not None:
            if not isinstance(seed, int):
                del params["seed"]
                if verbose:
                    print(f"Provided seed ({seed}) not int. Using default instead.")

    @staticmethod
    def _validate_rng(params: KeywordParams, verbose: bool = False) -> None:
        """Validate RNG is appropriate."""
        rng = params.get("rng")
        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                del params["rng"]
                if verbose:
                    print(
                        f"Provided rng ({rng}) not np.random.Generator. Using default instead."
                    )

    @staticmethod
    def _validate_volume_sampler(params: KeywordParams, verbose: bool = False) -> None:
        """Validate volume sampler is appropriate."""
        vs = params.get("volume_sampler")
        if vs:
            try:
                # test that it can be called with no input
                # test that the output is castable to int
                int(nonnegified(intified(vs))())
            except Exception as e:
                del params["volume_sampler"]
                if verbose:
                    print(
                        "Provided volume sampler cannot produce a value castable to int.\n"
                        + "default sampler will be used.\n"
                        + f"Error was {e}"
                    )
        else:
            vol = params.get("volume")
            if vol is not None:
                try:
                    int(vol)
                except Exception as e:
                    del params["volume_sampler"]
                    if verbose:
                        print(
                            "Provided volume is not castable to int.\n"
                            + "default sampler will be used.\n"
                            + f"Error was {e}"
                        )

    @staticmethod
    def _validate_ctrs(params: KeywordParams, verbose: bool = False) -> None:
        """Validate CTRs are appropriate."""
        floats = [
            "buyside_ctr",
            "buyside_ctr_avg",
            "sellside_paid_ctr",
            "sellside_paid_ctr_avg",
        ]
        for float_name in floats:
            f = params.get(float_name)
            if f is not None and not isinstance(f, float):
                del params[float_name]
                if verbose:
                    print(
                        f"Provided value for {float_name} is not float.\n"
                        + "A default will be used."
                    )

    @staticmethod
    def _validate_reward_distribution_sampler(
        params: KeywordParams, verbose: bool = False
    ) -> None:
        """Validate reward sampler is appropriate."""
        rds = params.get("reward_distribution_sampler")
        if rds:
            try:
                m, n = 2, 5
                assert len(rds(m)) == m and len(rds(n)) == n
                assert all([isinstance(f, float) for f in rds(n)])
            except Exception as e:
                if verbose:
                    print(e)
                    print(
                        f"debug_vrds_flt_err: {[isinstance(f, float) for f in rds(n)]}"
                    )
                del params["reward_distribution_sampler"]
        else:
            pts = params.get("reward_cdf_pts")
            if pts:
                if (
                    not isinstance(pts, List)
                    or len(pts) != 2
                    or (len(pts[0]) != len(pts[1]))
                    or not all([isinstance(f, float) for f in pts[0] + pts[1]])
                ):  # or check that things are sorted and in correct range of cdf in 0,1
                    del params["reward_cdf_pts"]

    @staticmethod
    def _validate_parameters(params: KeywordParams, verbose: bool = False) -> None:
        """Validate that all parameters are appropriate."""
        Keyword._validate_seed(params, verbose)
        Keyword._validate_rng(params, verbose)
        Keyword._validate_volume_sampler(params, verbose)
        Keyword._validate_ctrs(params, verbose)
        Keyword._validate_reward_distribution_sampler(params, verbose)

    @staticmethod
    def _rng_init(params: KeywordParams) -> np.random.Generator:
        """Initialize the RNG for the environment."""
        param_rng: Optional[np.random.Generator] = params.get("rng")
        if param_rng:
            return param_rng
        else:
            param_seed: int = params.get("seed", 1729)
            return np.random.default_rng(param_seed)

    @staticmethod
    def _volume_sampler_init(params: KeywordParams) -> Callable[[], int]:
        """Initialize the volume sampler."""
        # This is number of auctions in a given day; it can be random or fixed constant function.
        param_volume_sampler: Optional[Callable[[], int]] = params.get("volume_sampler")
        if param_volume_sampler:
            return param_volume_sampler
        else:
            # constant_volume_function
            def constant_volume() -> int:
                return nonnegify(int(params.get("volume", 1000)))

            return constant_volume

    @staticmethod
    def _buyside_ctr_init(rng: np.random.Generator, params: KeywordParams) -> float:
        """
        Initialize value for ad click rate | impression.

        If we observed a mean 0.045086 std 0.191016
        Using beta distribution with alpha of 1,
        we can get same mean with std of 0.20759835650309028 which is pretty close and has fat head.
        The pdf looks like 1/x.

        This is the distribution we sample a ctr from by default.
        If a different average value is provided,
        we'll sample from a beta with approximately that average for the default
        """
        param_buyside_ctr: Optional[float] = params.get("buyside_ctr")
        if param_buyside_ctr is not None:
            return probify(param_buyside_ctr)
        else:
            # return sample from a beta distribution to get a random ctr
            _buyside_ctr_avg: float = probify(params.get("buyside_ctr_avg", 0.045086))
            _buyside_ctr_beta: float = beta_param(_buyside_ctr_avg)
            return rng.beta(1.0, _buyside_ctr_beta)

    @staticmethod
    def _sellside_paid_ctr_init(
        rng: np.random.Generator, params: KeywordParams
    ) -> float:
        """
        Initialize value for paid click rate | buyside click.

        If we observed a mean 0.367151, std 0.468074
        Using a beta dist with alpha = 1,
        we can get same mean with std of 0.4997965582779267 which is pretty close.
        The pdf looks like sqrt function opening to the left from 1

        Note: Technically can have multiple paid clicks and each has own reward, but we're not modeling that.
        Instead we model one paid click="get reward" and a distribution of reward values for those
        """
        param_sellside_paid_ctr: Optional[float] = params.get("sellside_paid_ctr")
        if param_sellside_paid_ctr is not None:
            return probify(param_sellside_paid_ctr)
        else:
            _sellside_paid_ctr_avg: float = params.get(
                "sellside_paid_ctr_avg", 0.367151
            )  # given that a goog click has happened
            _sellside_paid_ctr_beta: float = beta_param(_sellside_paid_ctr_avg)
            return rng.beta(1.0, _sellside_paid_ctr_beta)

    @staticmethod
    def _reward_distribution_sampler_init(
        rng: np.random.Generator, params: KeywordParams
    ) -> Callable[[int], Iterable[float]]:
        """
        Initialize reward sampler.

        Reward is assumed to be independent of our bid for a given keyword.
        """
        param_reward_distribution_sampler: Optional[
            Callable[[int], Iterable[float]]
        ] = params.get("reward_distribution_sampler")
        if param_reward_distribution_sampler:
            return param_reward_distribution_sampler
        else:
            raise ValueError("Please provide a reward_distribution_sampler.")


class ExplicitKeyword(Keyword):
    """
    A Keyword with an explicit model of auction relationships.

    A Keyword with an explicit model of the bid->impression_share and bid->cost
    relationships for auctions rather than simulating other bidders.

    ...
    Additional Attributes:
        impression_rate (Callable[[float],float]): map of bid to impression likelihood
        cost_per_buyside_click (Callable[[float],float]): (probabilistic) map of bid to cost


    Methods
    -------
    sample_impressions(self, bid: float, num_auctions: int=1)->int:
        samples (int) number of auctions won out of maximum num_auctions for a given bid

    sample_buyside_costs(self, bid: float, n: int=1)->np.array:
        samples the cost for each of n impressions if they happen to be clicked

    auction(self, bid: float, num_auctions: int) -> Tuple[int, Iterable[int], Iterable[float]]:
        samples the number of impressions, placements, and individual buyside costs per click for num_auction many auctions. Does this with an explicit cost and impression model
    """

    def __init__(self, params: Dict[str, Any] = {}, verbose: bool = False) -> None:
        """Initialize samplers for the keyword."""
        super(ExplicitKeyword, self).__init__(params, verbose)
        self.impression_rate: Callable[
            [float], float
        ] = ExplicitKeyword._impression_rate_init(params)
        self.cost_per_buyside_click: Callable[
            [float], float
        ] = ExplicitKeyword._cost_per_buyside_click_init(self.rng, params)
        self.params = params

    def sample_impressions(self, bid: float, num_auctions: int = 1) -> int:
        """
        Sample impressions num_auctions times.

        Returns
            (int) sampled of the number of impressions seen for a given bid and volume
        """
        # TODO: Pass seed to this function.
        return rust.binomial_impressions(num_auctions, self.impression_rate(bid))

    def sample_buyside_costs(self, bid: float, n: int = 1) -> np.array:
        """
        Sample buyside costs n times.

        Returns:
            (float array) sampled buyside costs for each click at a fixed bid price

        Keyword arguments:
            bid (float): the max cost, and also the nput to the cpc sampler
            n (int): the length of the output array, and also the number of buyside clicks
        """
        if n < 1:
            return np.array([0])
        else:
            result = self.cost_per_buyside_click(bid, n)
            return result

    def auction(
        self, bid: float, num_auctions: int = 1
    ) -> Tuple[int, Iterable[int], Iterable[float]]:
        """
        Simulate results of num_auctions many auctions.

        Returns:
            impressions (int): number of impressions = nmumber of auctions won
            placements (Iterable[int]): location of each auction won
            costs (Iterable[float]): price of an adclick on each impression

        Arguments:
            bid (float): how much we bid in all the auctions
            num_auctions (int): the number of auctions run = max number of impressions
        """
        impressions = self.sample_impressions(bid, num_auctions=num_auctions)
        costs = self.sample_buyside_costs(bid, impressions)
        placements = np.zeros(costs.shape).astype(int)
        return impressions, placements, costs

    @staticmethod
    def _impression_rate_init(params: Dict[str, Any]) -> Callable[[float], float]:
        """
        Initialize the bid -> impression share model.

        Default options uses a sigmoid shape as this usually looks roughly like a
        sigmoid if we run an auction and ignore placement.
        """
        param_impression_rate = params.get("impression_rate")
        if param_impression_rate:
            return param_impression_rate
        else:

            def thresholded_sigmoid(x):
                return rust.threshold_sigmoid(x, params)

            return thresholded_sigmoid

    def _cost_per_buyside_click_init(
        rng: np.random.Generator, params: Dict[str, Any]
    ) -> Callable[[float], float]:
        """
        Initialize the cost of buyside adclick given a bid.

        constraints:
            0 <= cpc <= bid
            as bid -> 0, cpc -> bid
            as bid -> big, cpc -> high quantile of bid->impression func
            as bid -> big, cpc noisier
        TODO: don't love the default here...
        """
        param_cost_per_buyside_click = params.get("cost_per_buyside_click")
        if param_cost_per_buyside_click:
            return param_cost_per_buyside_click
        else:
            return rust.cost_create


class ImplicitKeyword(Keyword):
    """
    A Keyword with an implicit model of the bid->impression_share and bid->cost relationships for auctions.

    These are based on a simulation of an auction auction with other bidders.

    ...
    Additional Attributes:
        bidder_distribution (Callable[[], int]): can be sampled to get number of participating bidders in a given auction
        bid_distribution (Callable[[int,int], np.array]): can be sampled to get the bids of all the other participants in every auction. used for auction simulations


    Methods
    -------
    sample_bids(self, num_auctions: int=1)->np.array:
        samples (2D float array) each bid for each of num_auctions auctions

    auction(self, bid: float, num_auctions: int=1, n_winners: int=2) -> Tuple[int, Iterable[int], Iterable[float]]:
        samples the number of impressions, placements, and individual buyside costs per click for num_auction many auctions. Does this with a literal second price auction in which all the other bidders and their bids are simulated.
    """

    def __init__(self, params: Dict[str, Any] = {}, verbose: bool = False) -> None:
        """Initialize samplers for the keyword."""
        super(ImplicitKeyword, self).__init__(params, verbose)

        self.bidder_distribution: Callable[
            [], int
        ] = ImplicitKeyword._bidder_distribution_init(self.rng, params)
        self.bid_distribution: Callable[
            [int, int], np.array
        ] = ImplicitKeyword._bid_distribution_init(self.rng, params)

    def sample_bids(self, num_auctions: int = 1) -> np.array:
        """
        Sample bids from bid distribution num_auctions times.

        Returns:
            (2D float array) of each bid for each of num_auctions auctions

        Arguments:
            num_auctions (int): number of auctions
        """
        return self.bid_distribution(self.bidder_distribution(), num_auctions)
        # iffy: same num bidders in every sample if n > 1

    def auction(
        self, bid: float, num_auctions: int = 1, n_winners: int = 1
    ) -> Tuple[int, Iterable[int], Iterable[float]]:
        """
        Run a second price auction.

        Run a literal second price auction with n_winners where
        each winner's cost is the bid of the place below them.

        Returns:
            impressions (int): number of impressions = nmumber of auctions won
            placements (Iterable[int]): place of each auction won
            costs (Iterable[float]): price of an adclick on each impression

        Arguments:
            bid (float): how much we bid in all the auctions
            num_auctions (int): the number of auctions run = max number of impressions
            n_winners (int): number of winners in the second price auction. Higher values make it easier to win, and potentially lower costs.
        """
        # Implicit Keyword
        # num bidders x num_auction
        other_bids = self.sample_bids(num_auctions).T
        # run a 2nd price auction, n=1 would make it a first price auction.
        return nth_price_auction(bid, other_bids, n=2, num_winners=n_winners)

    @staticmethod
    def _bidder_distribution_init(
        rng: np.random.Generator, params: Dict[str, Any]
    ) -> Callable[[], int]:
        """Initialize the bidder distribution sampler."""
        param_bidder_distribution = params.get("bidder_distribution")
        if param_bidder_distribution:
            return param_bidder_distribution
        else:
            # default is binomial distribution of bidders
            # model bidders as a pool of identical bidders, and each one flips a coin. They bid if their coin comes up heads.
            max_bidders: int = params.get(
                "max_bidders", 30
            )  # tune this lower to add variance, and decrease winning bid threshold.
            participation_rate: float = probify(params.get("participation_rate", 3 / 5))

            def sample_binomial() -> int:
                return rng.binomial(max_bidders, participation_rate)

            return sample_binomial

    @staticmethod
    def _bid_distribution_init(
        rng: np.random.Generator, params: Dict[str, Any]
    ) -> Callable[[int, int], np.array]:
        """Initialize the bid distribution sampler."""
        param_bid_distribution = params.get("bid_distribution")

        if param_bid_distribution:
            return param_bid_distribution
        else:
            # laplace value of bids # based on nothing
            # for laplace dist this location param should basically be zero. it'll be lower than average winbid
            bid_loc: float = params.get("bid_loc", 0.0)
            # laplace distribution scale param. lower -> decrease variance (hugely) AND cost of win
            bid_scale: float = params.get("bid_scale", 0.1)

            def sample_laplacian(s: int, n: int) -> np.array:
                return rng.laplace(bid_loc, bid_scale, size=(s, n))

            return sample_laplacian  # iffy: same num bidders in every sample if n > 1
