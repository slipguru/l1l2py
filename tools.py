# -*- coding: utf-8 -*-
"""
Functions for normalization and Cholesky factorization
"""

from __future__ import division

import numpy as np
from numpy import linalg as la


def scale(X, center=True, normalize=True, axis=0):
    """ Center and/or Normalize columns or rows of the 2-dimensional input
        matrix.

    :Parameters:
        X : ndarray
            NxM 2-dimensional matrix

        center : boolean or ndarray
            If `True` the vectors on `axis` are centered.
            If is an array is considerered the mean array on `axis`. The
            dimension must agree

        normalize : boolean or ndarray
            If `True` the vector on `axis` are normalized (unit norm)
            If is an array is considerered the norm array on `axis`. The
            dimension must agree

        axis : int
            If 0 columns operations. If 1 rows operations.

    :Returns:
        - NxM 2-dimensional ndarray centered and/or normalized
        - ndarray with the means (calculated or passed)
        - ndarray with the normalizers (calculated or passed)
          (square-root of the sum of squares)

    :Note:
        If `center` is `False` the second ndarray return contains all zeros.
        If `normalize` is `False` the second ndarray return contains all
        ones.
        If one column (or row, if axis=1) contains all zeros, the column (or
        the row) is not normalized but returned as is (alla zeros).

    """
    N, M = X.shape

    if axis == 0:
        shp, rep = (1, M), N
    else:
        shp, rep = (N, 1), M

    if center is False:
        m = np.zeros(shp)
    else:
        if center is True:
            m = X.mean(axis=axis).reshape(shp)
        else:
            m = center

        X = X - np.repeat(m, rep, axis=axis)

    if normalize is False:
        n = np.ones(shp)
    else:
        if normalize is True:
            n = np.sqrt(np.sum(X*X, axis=axis)).reshape(shp)
            n[n == 0.0] = 1.0
        else:
            n = normalize

        X = X / np.repeat(n, rep, axis=axis)

    return X, m, n
