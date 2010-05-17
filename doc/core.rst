.. _core:

*************************************
Main functions (:class:`l1l2py`)
*************************************
.. currentmodule:: l1l2py

.. testsetup:: *

   import l1l2py
   import numpy


In this module, the two main stages of the :math:`\ell_1\ell_2` with double
optimization variable selection are implemented as in [DeMol09b]_.

Assume we are given a supervised training set :math:`(\mathbf{X}, \mathbf{Y})`,
we aim at selecting a linear model built on few relevant input variables with
good prediction ability.

The linear model is :math:`\mathbf{X}\boldsymbol{\beta}`, where
:math:`\boldsymbol{\beta}` is found as the minimizer of the (naive) elastic-net
functional combined with a regularized least squares functional.

Hence :math:`\boldsymbol{\beta}` depends on two regularization parameters,
:math:`\tau` and :math:`\lambda` and one correlation parameter :math:`\mu`.

The optimal solution is found in two different stages:

* **Stage I** (:func:`minimal_model`)

  This stage aims at selecting the optimal pair of regularization parameters
  :math:`\tau_{opt}` and :math:`\lambda_{opt}` within a k-fold cross validation
  loop for a fixed and small value of the correlation parameter :math:`\mu`.
       
  The function performs exactly the pesudocode described in
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


Stage I: Minimal Model Selection
================================
.. autofunction:: minimal_model

Stage II: Nested lists generation
=================================
.. autofunction:: nested_models

Complete model selection
========================
.. autofunction:: model_selection



