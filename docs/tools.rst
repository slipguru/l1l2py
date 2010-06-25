.. _tools:

***************************
Tools (:mod:`l1l2py.tools`)
***************************
.. currentmodule:: l1l2py.tools

.. testsetup:: *

   import l1l2py.tools
   import numpy


This module contains useful functions to be used in combination with the
main functions of the package.

The functions included in this module are divided in four groups:

* :ref:`range_generators`
* :ref:`data_normalizer`
* :ref:`error_functions`
* :ref:`cross_validation_utils`

.. _range_generators:

Range generators
================  

.. autofunction:: linear_range
.. autofunction:: geometric_range


.. rubric:: Note

The geometric sequence of :math:`n` elements
between :math:`a` and :math:`b` is

.. math::

    a,\ ar^1,\ ar^2,\ \dots,\ ar^{n-1}

where the ratio :math:`r` is

.. math::

    r = \left(\frac{b}{a}\right)^{\frac{1}{n-1}}


.. _data_normalizer:

Data normalizers 
================
.. autofunction:: center
.. autofunction:: standardize

.. _error_functions:

Error functions 
===============
.. autofunction:: regression_error
.. rubric:: Note

The regression error is calculated using the formula

.. math::

    error = \frac{\sum_{i=1}^N{| l_i - p_i|^2}} {N}
        \qquad
        l_i \in\ labels,\, p_i \in\ predicted


.. autofunction:: classification_error
.. rubric:: Note

The classification error is calculated using this formula

.. math::

    error = \frac{\sum_{i=1}^N{f(l_i, p_i)}}{N} \qquad
            l_i \in\ labels,\, p_i \in\ predictions,

where

.. math::

    f(l_i, p_i) =
    \left\{ 
        \begin{array}{l l}
          1 & \quad \text{if $sign(l_i) \neq sign(p_i)$}\\
          0 & \quad \text{otherwise}\\
        \end{array}
    \right.
    
.. warning::
    The classification error is calculated using the :func:`numpy.sign` function.
    Keep in mind that the ``sign(x)`` returns ``0 if x==0``.


.. autofunction:: balanced_classification_error
.. rubric:: Note

The balanced classification error is calculated using this formula:

.. math::

    error = \frac{\sum_{i=1}^N{w_i \cdot f(l_i, p_i)}}
                  {N} \qquad l_i \in\ labels,\, p_i \in\ predictions,

where :math:`f(l_i, p_i)` is as defined above.

With the default weigths the error function becomes:

.. math::

    error =
            \frac{\sum_{i=1}^N{|l_i - \overline{labels}| \cdot f(l_i, p_i)}}
                  {N}
            \qquad
            l_i \in\ labels,\, p_i \in\ predicted
    
.. warning::

    If ``labels`` contains only values belonging to **one** class,
    the functions returns always `0.0` because
    :math:`l_i - \overline{labels} = 0`, than :math:`w_i=0` for
    each :math:`i`.

.. _cross_validation_utils:

Cross Validation utilities
==========================
.. autofunction:: kfold_splits
.. autofunction:: stratified_kfold_splits

.. rubric:: Note

Running this functions more times with the same value of the parameter ``rseed``
gives **always** the same result, in order to allow repeatable experiments.
Note, moreover, that each of this functions sets the random seed equal
to ``None``, to restore a random seed for the following use of the
:mod:`random` module (see :func:`random.seed`).
