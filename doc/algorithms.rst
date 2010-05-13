.. _algorithms:

******************************************
Algorithms (:mod:`biolearning.algorithms`)
******************************************
.. currentmodule:: biolearning.algorithms
.. moduleauthor:: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
.. sectionauthor:: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>

.. testsetup:: *

   import biolearning.algorithms
   import numpy


Introduction
============
.. automodule:: biolearning.algorithms
    
RLS minimizes the following objective function:

.. math::

    \frac{1}{N} \| Y - X\beta \|_2^2 + \mu \|\beta\|_2^2

finding the optimal model :math:`\beta^*`, where :math:`X` is the ``data``
matrix and :math:`Y` contains the ``labels``.

:math:`\ell_1\ell_2` minimizes the following objective function:

.. math::

    \frac{1}{N} \| Y - X\beta \|_2^2 + \mu \|\beta\|_2^2 + \tau \|\beta\|_1

finding the optimal model :math:`\beta^*`, where :math:`X` is the ``data``
matrix and :math:`Y` contains the ``labels``.

The computation is iterative, each step updates the value of :math:`\beta`
until the convergence is reached ??:

.. math::

    \beta^{(k+1)} = \mathbf{S}_{\frac{\tau}{\sigma}} (
                        (1 - \frac{\mu}{\sigma})\beta^k +
                        \frac{1}{n\sigma}X^T[Y - X\beta^k]
                    )

where, :math:`\mathbf{S}_{\gamma > 0}` is the soft-thresholding function

    .. math::

    \mathbf{S}_{\gamma}(x) = sign(x) max(0, | x | - \frac{\gamma}{2})

Moreover, the function implements a *MFISTA* modification, wich
increases with quadratic factor the convergence rate of the algorithm.

The constant :math:`\sigma` is a (theorically optimal) step size wich
depends by the data:

.. math::

    \sigma = \frac{\|X^T X\|}{N} + \mu

The convergence is reached when:

.. math::

    \|\beta^k - \beta^{k-1}\| \leq \|\beta^k\| * tolerance

but the algorithm will be stop when the maximum number of iteration
is reached.

Regularization Algorithms
=========================
.. autofunction:: ridge_regression
.. autofunction:: l1l2_regularization

Utility Functions
=================
.. autofunction:: l1_bounds
.. autofunction:: l1l2_path
