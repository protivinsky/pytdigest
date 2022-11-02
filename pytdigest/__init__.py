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


from .pytdigest import TDigest

__all__ = ['TDigest']
__author__ = 'Tomas Protivinsky'
__version__ = "0.0.3"
