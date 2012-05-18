.. _l1l2py:

====================
L1L2Py Documentation
====================

:Release: |release|
:Homepage: http://slipguru.disi.unige.it/Software/L1L2Py
:Repository: https://bitbucket.org/slipguru/l1l2py

**L1L2Py** is a Python package which collects various implementations of
regularized (linear) models with the aim to offer a framework of optimization
algorithms that can be used interchangeably.

Current ``2.X`` version of L1L2Py is not compatible with the previous major
release ``1.X``. L1L2Py was born as a package able to solve a specific
(biological) feature-selection problem.
After the development and release of the
`L1L2Signature <http://slipguru.disi.unige.it/Software/L1L2Signature>`_ library,
L1L2Py only contains optimization/machine-learning tools L1L2Signature
(from ``1.X`` version) relies on.

A subset of models (or functionals) belonging to the wide class of regularized
models through a combination of :math:`\ell_1` and :math:`\ell_2` norm
penalties (or regularizers) are:

* Regularized Least Squares (or Ridge Regression)
* Lasso (or :math:`\ell_1`-regularized Least Squares)
* Elastic Net (or :math:`\ell_1\ell_2`-regularized Least Squares)
* Fused Lasso
* Total Variation
* Support Vector Machine
* ... and much more

L1L2Py makes use of `NumPy <http://numpy.scipy.org>`_ to provide fast
N-dimensional array manipulation and it is licensed under
`New BSD License <http://www.opensource.org/licenses/BSD-3-Clause>`_.

The library is divided into various submodules (that we aim to increase across
following releases), each one referring to a specific family of optimization
algorithms wrapped in L1L2Py :ref:`API <api>`-compliant python classes.

Quick Reference
===============
.. toctree::
   :maxdepth: 2

   api.rst 

.. tutorial.rst
.. core.rst
.. algorithms.rst
.. tools.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. Commented References!!!
.. .. rubric:: References

.. .. [Zou05]    H. Zou, T. Hastie,
              "Regularization and variable selection via the elastic net"
              J.R. Statist. Soc. B, 67 (2) pp. 301-320, 2005
.. .. [DeMol09a] C. De Mol, E. De Vito, L. Rosasco,
              "Elastic-net regularization in learning theory"
              Journal of Complexity, n. 2, vol. 25, pp. 201-230, 2009.
.. .. [DeMol09b] C. De Mol, S. Mosci, M. Traskine, A. Verri,
              "A Regularized Method for Selecting Nested Group of Genes from
              Microarray Data"
              Journal of Computational Biology, vol. 16, pp. 677-690, 2009.
.. .. [Beck09]   A. Beck, M. Teboulle,
              "A fast iterative shrinkage-thresholding algorithm for linear
              inverse problems"
              SIAM Journal on Imaging Sciences, 2(1):183–202, Mar 2009.
