.. _tools:

*********************************
Tools (:mod:`l1l2py.tools`)
*********************************
.. currentmodule:: l1l2py.tools
.. moduleauthor:: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
.. sectionauthor:: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>

.. testsetup:: *

   import l1l2py.tools
   import numpy

Introduction
============
.. automodule:: l1l2py.tools

Range generators
================
The geometric sequence of :math:`n` elements
between :math:`a` and :math:`b` is

.. math::

    a,\ ar^1,\ ar^2,\ \dots,\ ar^{n-1}

where the ratio :math:`r` is

.. math::

    r = \left(\frac{b}{a}\right)^{\frac{1}{n-1}}
    
.. autofunction:: linear_range
.. autofunction:: geometric_range

Data Normalization 
==================
.. autofunction:: center
.. autofunction:: standardize

Error calculation 
=================
The classification error is calculated using this formula

.. math::

    error = \frac{\sum_{i=1}^N{f(l_i, p_i)}}{N} \qquad
            l_i \in\ labels,\, p_i \in\ predicted

.. math::

    f(l_i, p_i)=1 \quad if sign(l_i) \neq sign(p_i)

    f(l_i, p_i)=0 \quad otherwise
    
The balanced classification error is calculated using this formula

.. math::

    error = \frac{\sum_{i=1}^N{w_i \cdot f(l_i, p_i)}}
                  {N}
          =
            \frac{\sum_{i=1}^N{|l_i - \overline{labels}| \cdot f(l_i, p_i)}}
                  {N}
            \qquad
            l_i \in\ labels,\, p_i \in\ predicted

where

.. math::

    f(l_i, p_i)=1 \quad if sign(l_i) \neq sign(p_i)

    f(l_i, p_i)=0 \quad otherwise
    
.. warning::

    If ``labels`` contains only values belonging to **one** class,
    the functions returns always `0.0` because
    :math:`l_i - \overline{labels} = 0`, than :math:`w_i=0` for
    each :math:`i`.

The regression error is calculated using the formula

.. math::

    error = \frac{\sum_{i=1}^N{|l_i - p_i|^2}} {N}
        \qquad
        l_i \in\ labels,\, p_i \in\ predicted
            

.. autofunction:: classification_error
.. autofunction:: balanced_classification_error
.. autofunction:: regression_error

Cross Validation utilities
==========================
.. autofunction:: kfold_splits
.. autofunction:: stratified_kfold_splits
