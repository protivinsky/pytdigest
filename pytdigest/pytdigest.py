#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Apache License, Version 2.0,
# http://www.apache.org/licenses/LICENSE-2.0
#
# Copyright (c) 2015 Ted Dunning, All rights reserved.
#      https://github.com/tdunning/t-digest
# Copyright (c) 2018 Andrew Werner, All rights reserved.
#      https://github.com/ajwerner/tdigestc
# Copyright (c) 2022 Tomas Protivinsky, All rights reserved.
#      https://github.com/protivinsky/pytdigest


from __future__ import annotations
import numpy as np
from numpy.ctypeslib import ndpointer
import pandas as pd
import os
import ctypes
from numbers import Number
from typing import Union, List, Iterable, Optional


_path = os.path.dirname(os.path.realpath(__file__))
_lib = ctypes.CDLL(os.path.join(_path, 'tdigest.so'))
# _tdigest_dll = ctypes.CDLL('tdigest.so')

class _Centroid(ctypes.Structure):
    _fields_ = [("mean", ctypes.c_double), ("weight", ctypes.c_double)]

class _TDigest(ctypes.Structure):
    _fields_ = [
        ("delta", ctypes.c_double),
        ("max_centroids", ctypes.c_int),
        ("num_merged", ctypes.c_int),
        ("num_unmerged", ctypes.c_int),
        ("merged_weight", ctypes.c_double),
        ("unmerged_weight", ctypes.c_double),
        ("centroids", _Centroid * 0)
    ]


# td_new
_lib.td_new.argtypes = [ctypes.c_double]
_lib.td_new.restype = ctypes.POINTER(_TDigest)
# td_free
_lib.td_free.argtypes = [ctypes.POINTER(_TDigest)]
# td_add
_lib.td_add.argtypes = [ctypes.POINTER(_TDigest), ctypes.c_double, ctypes.c_double]
_lib.td_add_batch.argtypes = [
    ctypes.POINTER(_TDigest),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]

_lib.td_quantile_of.argtypes = [ctypes.POINTER(_TDigest), ctypes.c_double]
_lib.td_quantile_of.restype = ctypes.c_double
_lib.td_cdf_batch.argtypes = [
    ctypes.POINTER(_TDigest),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]

_lib.td_value_at.argtypes = [ctypes.POINTER(_TDigest), ctypes.c_double]
_lib.td_value_at.restype = ctypes.c_double
_lib.td_inverse_cdf_batch.argtypes = [
    ctypes.POINTER(_TDigest),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]

# _tdigest_dll.td_copy.argtypes = [ctypes.POINTER(_TDigest)]
# _tdigest_dll.td_copy.restype = ctypes.POINTER(_TDigest)

_lib.td_merge.argtypes = [ctypes.POINTER(_TDigest), ctypes.POINTER(_TDigest)]

_lib.td_total_weight.argtypes = [ctypes.POINTER(_TDigest)]
_lib.td_total_weight.restype = ctypes.c_double
_lib.td_total_sum.argtypes = [ctypes.POINTER(_TDigest)]
_lib.td_total_sum.restype = ctypes.c_double

_lib.merge.argtypes = [ctypes.POINTER(_TDigest)]

_lib.td_get_centroid.argtypes = [ctypes.POINTER(_TDigest), ctypes.c_int]
_lib.td_get_centroid.restype = ctypes.POINTER(_Centroid)
_lib.td_get_centroids.argtypes = [
    ctypes.POINTER(_TDigest),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]

_lib.td_of_centroids.argtypes = [
    ctypes.c_double,
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]
_lib.td_of_centroids.restype = ctypes.POINTER(_TDigest)

_lib.td_fill_centroids.argtypes = [
    ctypes.POINTER(_TDigest),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
]


class TDigest:

    def __init__(self, compression=100):
        self.compression = compression
        self._tdigest = _lib.td_new(compression)

    def __del__(self):
        _lib.td_free(self._tdigest)

    def update(self, x, w=None, raise_if_nans: bool = False):
        x = TDigest._unwrap_if_possible(x)
        w = TDigest._unwrap_if_possible(w)
        if isinstance(x, Number):
            if np.isfinite(x):
                if w is None:
                    _lib.td_add(self._tdigest, float(x), 1.)
                elif np.isfinite(w):
                    _lib.td_add(self._tdigest, float(x), w)
                elif raise_if_nans:
                    raise ValueError("Weight has an invalid value.")
            elif raise_if_nans:
                raise ValueError("Value is invalid.")
        elif isinstance(x, np.ndarray):
            # TODO: make this robust to nans and infinities
            size = x.size
            if w is None:
                w = np.ones_like(x)
            elif isinstance(w, np.ndarray):
                if w.size != size:
                    raise TypeError("w has to be of the same size as values.")
            else:
                raise TypeError("Weights are unrecognized type.")
            invalid = np.isnan(x) | np.isinf(x) | np.isnan(w) | np.isinf(w) | (w < 0)
            if raise_if_nans and np.sum(invalid):
                raise ValueError('x or w contains invalid values (nan or infinity).')
            else:
                x = x[~invalid]
                w = w[~invalid]
                if not x.flags.c_contiguous:
                    x = x.copy()
                if not w.flags.c_contiguous:
                    w = w.copy()
                _lib.td_add_batch(self._tdigest, size, x, w)
        else:
            raise TypeError("Values are unrecognized type.")

    def cdf(self, at: Union[Number, List, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(at, list):
            at = np.array(at)
        if isinstance(at, Number):
            return _lib.td_quantile_of(self._tdigest, at)
        elif isinstance(at, np.ndarray):
            if at.ndim > 1:
                raise "at parameter cannot be a multidimensional array."
            quantiles = np.empty_like(at)
            _lib.td_cdf_batch(self._tdigest, at.size, at, quantiles)
            return quantiles
        else:
            raise TypeError("At parameter is unrecognized type.")

    def inverse_cdf(self, quantile: Union[Number, List, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(quantile, list):
            quantile = np.array(quantile)
        if isinstance(quantile, Number):
            return _lib.td_value_at(self._tdigest, quantile)
        elif isinstance(quantile, np.ndarray):
            if quantile.ndim > 1:
                raise "Quantile cannot be a multidimensional array."
            values = np.empty_like(quantile)
            _lib.td_inverse_cdf_batch(self._tdigest, quantile.size, quantile, values)
            return values
        else:
            raise TypeError("Quantile is unrecognized type.")

    def __iadd__(self, other):
        if not isinstance(other, TDigest):
            raise "Only TDigest can be add to TDigest."
        else:
            _lib.td_merge(self._tdigest, other._tdigest)
            return self

    def __add__(self, other):
        result = self.__copy__()
        result += other
        return result

    def __copy__(self):
        other = TDigest(self.compression)
        other += self
        return other

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def __str__(self) -> str:
        centroids = self._num_merged + self._num_unmerged
        merged = 'not merged' if self._num_unmerged else 'merged'
        return f'TDigest(mean={self.mean:.3g}, weight={self.weight:.3g}, ' \
               f'centroids={centroids}, {merged}, compression={self.compression})'

    def __repr__(self) -> str:
        return self.__str__()

    def get_centroid(self, i):
        if i >= self._num_merged + self._num_unmerged:
            raise IndexError(f'Cannot access centroid at index {i}, TDigest has only'
                             f' {self._num_merged + self._num_unmerged} centroids.')
        centroid = _lib.td_get_centroid(self._tdigest, i)
        return centroid.contents.mean, centroid.contents.weight

    def get_centroids(self):
        self.force_merge()
        centroids = np.empty(2 * self._num_merged, dtype='float')
        _lib.td_get_centroids(self._tdigest, centroids)
        return centroids.reshape([-1, 2])

    def force_merge(self):
        _lib.merge(self._tdigest)

    # @staticmethod
    # def of_centroids(centroids: np.ndarray, compression: float = 100):
    #     if centroids.ndim != 2 or centroids.shape[1] != 2:
    #         raise TypeError('Centroids have to be 2-dimensional np.array with 2 columns (means and weights)')
    #     if centroids.shape[0] > 6 * compression + 10:
    #         raise TypeError(f'Num of centroids {centroids.shape[0]} is too large for TDigest with '
    #                         f'compression {compression}.')
    #     elif centroids.shape[0] > compression:
    #         print(f'Num of centroids seems large.')
    #     _tdigest = _tdigest_dll.td_of_centroids(compression, centroids.shape[0], centroids.reshape(-1))
    #     td = TDigest(compression)
    #     _tdigest_dll.td_free(td._tdigest)
    #     td._tdigest = _tdigest
    #     return td

    @staticmethod
    def of_centroids(centroids: np.ndarray, compression: float = 100):
        if centroids.ndim != 2 or centroids.shape[1] != 2:
            raise TypeError('Centroids have to be 2-dimensional np.array with 2 columns (means and weights)')
        if centroids.shape[0] > 6 * compression + 10:
            raise TypeError(f'Num of centroids {centroids.shape[0]} is too large for TDigest with '
                            f'compression {compression}.')
        elif centroids.shape[0] > compression:
            print(f'Num of centroids seems large.')
        td = TDigest(compression)
        _lib.td_fill_centroids(td._tdigest, centroids.shape[0], centroids.reshape(-1))
        return td

    @staticmethod
    def combine(first: Union[TDigest, Iterable[TDigest]], second: Optional[TDigest] = None) -> TDigest:
        if second is None:
            if isinstance(first, pd.Series):
                first = first.values
            result = None
            for other in first:
                if result is None:
                    result = other.__copy__()
                else:
                    result += other
            return result
        else:
            if not (isinstance(first, TDigest) and isinstance(second, TDigest)):
                raise TypeError(f'Both first and second arguments have to be instances of TDigest.')
            return first + second

    @property
    def weight(self):
        return _lib.td_total_weight(self._tdigest)

    @property
    def mean(self):
        if self.weight == 0:
            return np.nan
        else:
            return _lib.td_total_sum(self._tdigest) / self.weight

    @property
    def _delta(self):
        return self._tdigest.contents.delta

    @property
    def _max_centroids(self):
        return self._tdigest.contents.max_centroids

    @property
    def _num_merged(self):
        return self._tdigest.contents.num_merged

    @property
    def _num_unmerged(self):
        return self._tdigest.contents.num_unmerged

    @property
    def _merged_weight(self):
        return self._tdigest.contents.merged_weight

    @property
    def _unmerged_weight(self):
        return self._tdigest.contents.unmerged_weight

    @staticmethod
    def _unwrap_if_possible(x: Union[Number, np.ndarray, pd.Series]) -> Union[Number, np.ndarray]:
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(x, np.ndarray) and x.size == 1:
            x = float(x)
        return x





