.. _core:

*************************************
Main functions (:class:`l1l2py`)
*************************************
.. currentmodule:: l1l2py

.. testsetup:: *

   import l1l2py
   import numpy


This module implements the two main stages of the :math:`\ell_1\ell_2` with double
optimization variable selection, as in [DeMol09b]_.

Given a supervised training set :math:`(\mathbf{X}, \mathbf{Y})`,
the aim is to select a linear model built on few relevant input variables with
good prediction ability.

The linear model is :math:`\mathbf{X}\boldsymbol{\beta}`, where
:math:`\boldsymbol{\beta}` is found as the minimizer of the (naive) elastic-net
functional combined with a regularized least squares functional.

.. math::
    \frac{1}{n} \| \mathbf{Y} - \mathbf{X}\boldsymbol{\beta} \|_2^2
    + \mu \|\boldsymbol{\beta}\|_2^2
    + \tau \|\boldsymbol{\beta}\|_1

.. math::
    \frac{1}{n} \| \mathbf{Y} - \mathbf{\tilde{X}}\boldsymbol{\tilde{\beta}} \|_2^2
    + \lambda \|\boldsymbol{\tilde{\beta}}\|_2^2

in which :math:`\boldsymbol{\tilde{\beta}}` and :math:`\mathbf{\tilde{X}}`
represent, respectively, the weights vector and the input matrix restricted to
the genes selected by the :math:`\ell_1\ell_2` selection.

The optimal solution depends on two regularization parameters, :math:`\tau` and
:math:`\lambda` and one correlation parameter :math:`\mu` and is found in
two different stages:

* **Stage I** (:func:`minimal_model`)

  This stage aims at selecting the optimal pair of regularization parameters
  :math:`\tau_{opt}` and :math:`\lambda_{opt}` within a k-fold cross validation
  loop for a fixed and small value of the correlation parameter :math:`\mu`.

  The function follows exactly the pesudocode described in
  [DeMol09b]_ (pag.7 - Stage I).


* **Stage II** (:func:`nested_models`)

  For fixed :math:`\tau_{opt}` and :math:`\lambda_{opt}`, Stage II identifies
  the set of relevant lists of variables for increasing values of the correlation
  parameter :math:`\mu`.

  .. note:: For :math:`\tau_{opt}` and :math:`\lambda_{opt}` the lists of
            relevant variables have same prediction power [DeMol09a]_.

  The function performs exactly the pesudocode described in
  [DeMol09b]_  (pag.7 - Stage II).


This module also provide a wrapper function (:func:`model_selection`) that
runs the two stages sequentially.

.. _stage_i:

Stage I: Minimal Model Selection
================================
.. autofunction:: minimal_model

.. _stage_ii:

Stage II: Nested lists generation
=================================
.. autofunction:: nested_models

Complete model selection
========================
.. autofunction:: model_selection
