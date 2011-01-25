.. _algorithms:

*************************************
Algorithms (:mod:`l1l2py.algorithms`)
*************************************
.. currentmodule:: l1l2py.algorithms

.. testsetup:: *

   import l1l2py.algorithms
   import numpy


In order to describe the function implemented in this module, we have to assume
some notation.

Assuming to have a centered data matrix
:math:`\mathbf{X} \in \mathbb{R}^{n \times p}` and a column vector of
regression values :math:`\mathbf{Y} \in \mathbb{R}^n` or binary labels
:math:`\mathbf{Y} \in \{-1, 1\}^n`, we want to minimize the
regression/classification error solving a Regularized Least Square (RLS)
problem.

In this module two main algorithms are implemented. The first one solves a
classical RLS problem with a penalty on the :math:`\ell_2\text{-norm}`
of the vector :math:`\boldsymbol{\beta}` (also called :func:`ridge_regression`)

.. math::
    \boldsymbol{\beta^*} =
        \argmin_{\boldsymbol{\beta}}
            \Big\{
            \frac{1}{n} \| \mathbf{Y} - \mathbf{X}\boldsymbol{\beta} \|_2^2
            + \mu \|\boldsymbol{\beta}\|_2^2
            \Big\},
    :label: rls

with :math:`\mu > 0`.

The second one minimizes a functional with a linear combination of two penalties
on the :math:`\ell_1\text{-norm}` and :math:`\ell_2\text{-norm}` of the vector
:math:`\boldsymbol{\beta}` (also called :func:`l1l2_regularization`)

.. math::
    \boldsymbol{\beta^*} =
        \argmin_{\boldsymbol{\beta}}
            \Big\{
            \frac{1}{n} \| \mathbf{Y} - \mathbf{X}\boldsymbol{\beta} \|_2^2
            + \mu \|\boldsymbol{\beta}\|_2^2
            + \tau \|\boldsymbol{\beta}\|_1
            \Big\},
    :label: l1l2

with :math:`\mu > 0` and :math:`\tau > 0`.


Implementation details
======================
While :eq:`rls` has closed-form solution, for :eq:`l1l2` there are many
different approaches. In this module we provide an Iterative
Shrinkage-Thresholding Algorithm (ISTA) proposed in [DeMol09a]_ exploiting
a faster variation (called FISTA) proposed in [Beck09]_.

Starting from a null vector :math:`\boldsymbol{\beta}`, each step updates
the value of :math:`\boldsymbol{\beta}` until convergence:

.. math::
    \boldsymbol{\beta}^{(k+1)} =
        \mathbf{S}_{\frac{\tau}{\sigma}} (
                        (1 - \frac{\mu}{\sigma})\boldsymbol{\beta}^k +
                        \frac{1}{n\sigma}
                            \mathbf{X^T}[\mathbf{Y} -
                                         \mathbf{X}\boldsymbol{\beta}^k]
                    )

where, :math:`\mathbf{S}_{\gamma > 0}` is the soft-thresholding function

.. math::
    \mathbf{S}_{\gamma}(x) = \text{sign}(x) \text{max}(0, | x | - \gamma/2)

The constant :math:`\sigma` is a (theorically optimal) step size which
depends on the data:

.. math::
    \sigma = \frac{e}{n} + \mu,

where :math:`e` is the maximum eigenvalue of the matrix
:math:`\mathbf{X^T}\mathbf{X}`.

The convergence is reached when for each :math:`j \in \{0,\dots,d-1\}`:

.. math::
    | \beta_j^k - \beta_j^{k-1} | \leq | \beta_j^k | * (tol/k),

where :math:`tol > 0` and before :math:`k` reaches a fixed maximum number of
iterations.

Regularization Algorithms
=========================
.. autofunction:: ridge_regression
.. autofunction:: l1l2_regularization

Utility Functions
=================
.. autofunction:: l1_bound
.. autofunction:: l1l2_path

.. rubric:: Note

The acceleration method based on *warm starts*, implemented in this function,
is been theoretically proved in [Hale08]_.

.. [Hale08] E. T. Hale, W. Yin, Y. Zhang
            "Fixed-point continuation for :math:`\ell_1`-minimization:
            Methodology and convergence"
            SIAM J. Optim. Volume 19, Issue 3, pp. 1107-1130, 2008
