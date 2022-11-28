#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTDigest
=========

Python package for *fast* TDigest calculation.

- C implementation and thin Python wrapper
- suitable for big data and streaming and distributed settings
- developed for smooth compatibility with Pandas and numpy

Based on previous work of Ted Dunning and Andrew Werner.

Basic example
-------------

.. code:: python

    from pytdigest import TDigest
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(12354)
    n = 100_000
    x = rng.normal(loc=0, scale=10, size=n)
    w = rng.exponential(scale=1, size=n)

    # estimation from data is simple:
    td = TDigest.compute(x, w, compression=1_000)
    # td now contains "compressed" distribution: centroids with their means and weights

    # TDigest can be used to provide mean or approximate quantiles (i.e. inverse CDF):
    td.mean
    quantiles = [0., 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.]
    td.inverse_cdf(quantiles)

    # these results are close to numpy values (note that numpy provides only unweighted quantiles)
    np.average(x, weights=w)  # mean should be exact
    np.quantile(x, quantiles)

    # TDigest can be copied
    td2 = td.copy()

    # and multiple TDigests can be added together to provide approximate quantiles for the overall dataset
    td + td2

Legal stuff
-----------

Apache License, Version 2.0,
http://www.apache.org/licenses/LICENSE-2.0

Copyright (c) 2015 Ted Dunning, All rights reserved.
     https://github.com/tdunning/t-digest
Copyright (c) 2018 Andrew Werner, All rights reserved.
     https://github.com/ajwerner/tdigestc
Copyright (c) 2022 Tomas Protivinsky, All rights reserved.
     https://github.com/protivinsky/pytdigest
"""


from .pytdigest import TDigest, HandlingInvalid

__all__ = ['TDigest', 'HandlingInvalid']
__author__ = 'Tomas Protivinsky'
__version__ = "0.0.6"
