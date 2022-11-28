#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import math
from copy import copy
from statsmodels.stats.weightstats import DescrStatsW
from pytdigest import TDigest, HandlingInvalid

# TODO:
# how to do tests if I need to compile the plugin?
# will GitHub actions handle it?
# and is there any sensible way how to run the tests locally? what can I do on Windows?


def test_basic_operations():
    td = TDigest.compute(100, 10)
    # TDigest is constructed and has the correct properties
    assert math.isclose(td.mean, 100)
    assert math.isclose(td.weight, 10)
    td.update(0, 10)
    # update works as expected
    assert math.isclose(td.mean, 50)
    assert math.isclose(td.weight, 20)
    td2 = copy(td)
    td += td2
    # ensure that addition works and copy produces different objects
    assert math.isclose(td.mean, 50)
    assert math.isclose(td.weight, 40)
    assert math.isclose(td2.weight, 20)
    assert id(td) != id(td2)
    td.scale_weight(0.5)
    # scale_weight works
    assert math.isclose(td.mean, 50)
    assert math.isclose(td.weight, 20)


def test_to_centroids_and_back():
    rng = np.random.Generator(np.random.PCG64(12345))
    xarr = rng.normal(loc=100, scale=20, size=1000)
    warr = rng.normal(loc=10, scale=2, size=1000)
    td = TDigest.compute(xarr, warr)
    cs = td.get_centroids()
    td2 = TDigest.of_centroids(cs, compression=td.compression)
    assert math.isclose(td.mean, td2.mean)
    assert math.isclose(td.weight, td2.weight)
    for i in range(td._num_merged + td._num_unmerged):
        c1 = td.get_centroid(i)
        c2 = td2.get_centroid(i)
        assert math.isclose(c1[0], c2[0])
        assert math.isclose(c1[1], c2[1])
    quantiles = [0., 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.]
    qs1 = td.inverse_cdf(quantiles)
    qs2 = td2.inverse_cdf(quantiles)
    for q1, q2 in zip(qs1, qs2):
        assert math.isclose(q1, q2)


def test_correct_calculation():
    rng = np.random.Generator(np.random.PCG64(12345))
    n = 100_000
    xarr = rng.normal(loc=100, scale=20, size=n)
    quantiles = [0., 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.]
    np_qs = np.quantile(xarr, quantiles)
    np_mean = np.mean(xarr)
    td = TDigest.compute(xarr, compression=1_000)
    td_qs = td.inverse_cdf(quantiles)
    assert math.isclose(np_mean, td.mean)
    for q1, q2 in zip(np_qs, td_qs):
        # TDigest quantiles are only approximate, so need higher tolerance
        assert math.isclose(q1, q2, rel_tol=1e-3)


def test_correct_weighted_calculation():
    rng = np.random.Generator(np.random.PCG64(12345))
    n = 100_000
    xarr = rng.normal(loc=100, scale=20, size=n)
    warr = rng.normal(loc=10, scale=2, size=n)
    td = TDigest.compute(xarr, warr, compression=1_000)
    dsw = DescrStatsW(xarr, weights=warr)
    assert math.isclose(td.mean, dsw.mean)
    quantiles = [0., 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.]
    td_qs = td.inverse_cdf(quantiles)
    dsw_qs = dsw.quantile(quantiles)  # [dsw.quantile(q) for q in quantiles]
    for q1, q2 in zip(dsw_qs, td_qs):
        # might need to increase the tolerance here
        assert math.isclose(q1, q2, rel_tol=1e-3)


def test_cdf():
    rng = np.random.Generator(np.random.PCG64(12345))
    xarr = rng.normal(loc=0, scale=1, size=100_000)
    td = TDigest.compute(xarr, compression=1_000)
    # TODO: verify that you get the correct normal cdf



@pytest.fixture
def df():
    rng = np.random.Generator(np.random.PCG64(12345))
    n = 100_000
    g = rng.integers(low=1, high=11, size=n)
    x = 10 * g + rng.normal(loc=0, scale=50, size=n)
    df = pd.DataFrame({'g': g, 'x': x, 'w': g})
    return df


def test_combine(df):
    td_all = TDigest.compute(df['x'], df['w'], compression=1_000)
    td_per_group = df.groupby('g').apply(lambda frame: TDigest.compute(frame['x'], frame['w'], compression=1_000))
    td_from_groups = TDigest.combine(td_per_group)
    # TODO: might need to increase the tolerance as the quantiles are approximate only
    assert math.isclose(td_all.mean, td_from_groups.mean)
    assert math.isclose(td_all.weight, td_from_groups.weight)
    quantiles = [0., 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.]
    qs1 = td_all.inverse_cdf(quantiles)
    qs2 = td_from_groups.inverse_cdf(quantiles)
    for q1, q2 in zip(qs1, qs2):
        assert math.isclose(q1, q2, rel_tol=5e-3)
