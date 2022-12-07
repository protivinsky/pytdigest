#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python package for **fast** TDigest calculation.

- C implementation and thin Python wrapper
- suitable for big data and streaming and distributed settings
- developed for smooth compatibility with Pandas and numpy

Based on previous work of Ted Dunning and Andrew Werner.

- https://github.com/tdunning/t-digest
- https://github.com/ajwerner/tdigestc

.. warning::
    The package relies on C implementation that is compiled on host machine. I tested the compilation on Linux and
    on Windows 11, but I cannot ensure the compatibility with all operating systems. Let me know if you have encountered
    any issues.

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

Performance
-----------

Ted Dunning's original algorithm in Java takes about ~140 ns per addition on average. This package needs about ~200 ns
per addition when called from Python on numpy arrays, so the performance is fairly comparable with the original
implementation. All other tested TDigest implementations in Python are much slower.

.. code:: python

    import numpy as np
    from pytdigest import TDigest
    import time

    rng = np.random.Generator(np.random.PCG64(12345))

    for n in [100_000, 1_000_000, 10_000_000]:
        x = rng.normal(size=n)
        w = rng.exponential(size=n)

        start = time.time()
        td = TDigest.compute(x, w)
        end = time.time()
        print(f'PyTDigest, n = {n:,}: {end - start:.3g} seconds')

    # PyTDigest, n = 100,000: 0.0222 seconds
    # PyTDigest, n = 1,000,000: 0.21 seconds
    # PyTDigest, n = 10,000,000: 2.02 seconds

Similar packages
----------------

Several Python packages or wrappers exist for the TDigest algorithm.

tdigest
.......

The most popular on GitHub is a pure Python
`tdigest package
<https://github.com/CamDavidsonPilon/tdigest>`_. Pure Python implementation is indeed very slow – more than 100x
slower than this package:

.. code:: python

    import numpy as np
    from pytdigest import TDigest
    from tdigest import TDigest as TDigestPython
    import time

    rng = np.random.Generator(np.random.PCG64(12345))
    n = 100_000
    x = rng.normal(size=n)
    w = rng.exponential(size=n)

    start = time.time()
    td = TDigest.compute(x, w)
    end = time.time()
    print(f'PyTDigest: {end - start:.3g} seconds')
    # PyTDigest: 0.0246 seconds

    tdp = TDigestPython()
    start = time.time()
    tdp.batch_update(x)
    end = time.time()
    print(f'TDigest: {end - start:.3g} seconds')
    # TDigest: 7.26 seconds

Different weights for can be used in tdigest only with `update` method for adding a single observation.

t-digest CFFI
.............

Other package is `t-digest CFFI
<https://github.com/kpdemetriou/tdigest-cffi>`_, a thin Python wrapper over C implementation. It does not pass
batch updates into the C layer, so the iteration has to be done in python:

.. code:: python

    import numpy as np
    from tdigest import TDigest as TDigestCFFI
    import time

    rng = np.random.Generator(np.random.PCG64(12345))
    n = 100_000
    x = rng.normal(size=n)

    tdcffi = TDigestCFFI()
    start = time.time()
    for xx in x:
        tdcffi.insert(xx)
    end = time.time()
    print(f'TDigest-CFFI: {end - start:.3g} seconds')

Hence, this package is still almost 20x slower than this package when used over numpy arrays. In addition, t-digest CFFI
package allows only for integer weights.

qtdigest
........

`qtdigest
<https://github.com/QunarOPS/qtdigest>`_'s own benchmarking states that 100 000 additions take about 1.7 s, so it is
again almost 100x slower than this package.

tdigestc
........

`tdigestc
<https://github.com/ajwerner/tdigestc>`_ by ajwerner is a simple C implementation with wrappers for different
languages. The Python wrapper is very basic, it is not published on PyPI and some functionality was missing
in the underlying C implementation (for instance support for batch updates based on numpy arrays), so I took this
package as the starting point and added several useful features for use as a standalone Python package.

Future plans
------------

There are several improvements that can be done in the future:

- TDigest can calculate exact variance in addition to mean.
- Alternating merging procedure (the centroids are always merged left to right in the C implementation,
  however Ted Dunning states that alternating merging improves the precision).
- Scaling function for merging centroids is hard-coded at the moment. Ted Dunning mentions several
  possible functions that can be used in merging.
- Centroids can store information about their variance - the resulting TDigest should be still
  composable and fast and it can work much better for discrete distributions.

Documentation
-------------

- https://protivinsky.github.io/pytdigest/index.html

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
__version__ = '0.1.3'
