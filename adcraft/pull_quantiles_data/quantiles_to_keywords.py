"""Pull prod quantiles into csvs and sample values from them."""
import numpy as np
import pandas as pd


def floatify_df(data: pd.DataFrame) -> pd.DataFrame:
    """Convert all df fields to floats."""
    for c in data.columns:
        data[c] = data[c].astype(float)
    return data


def sample_from_quantiles(n, num_buckets, mins, meds, maxs, rng):
    """Return a list of values sampled from given quantile buckets.

    First, the bucket/quantile to sample from is uniformly sampled.
    Then, within each bucket, values are sampled uniformly from a
    linear interpolation between the min, median, and max for that bucket.
    """
    out = []
    buckets = rng.integers(low=0, high=num_buckets, size=(n,))
    samples = rng.random(size=(n,))
    # print(buckets)
    for bucket, q in zip(buckets, samples):
        out.append(
            np.interp(q, [0.0, 0.5, 1.0], [mins[bucket], meds[bucket], maxs[bucket]])
        )
    return out
