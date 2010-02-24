.. _tools:

*********************************
Tools (:mod:`biolearning.tools`)
*********************************
.. currentmodule:: biolearning.tools
.. moduleauthor:: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>
.. sectionauthor:: Salvatore Masecchia <salvatore.masecchia@disi.unige.it>

.. testsetup:: *

   import biolearning.tools
   import numpy

.. automodule:: biolearning.tools

Range generators
----------------
.. autofunction:: linear_range
.. autofunction:: geometric_range

Data Normalization 
------------------
.. autofunction:: center
.. autofunction:: standardize

Error calculation 
-----------------
.. autofunction:: classification_error
.. autofunction:: balanced_classification_error
.. autofunction:: regression_error

Cross Validation utilities
--------------------------
.. warning::
    
    The following functions are simple wrappers on similar function of the
    `mlpy <https://mlpy.fbk.eu/>`_  library.
    
.. autofunction:: kfold_splits
.. autofunction:: stratified_kfold_splits
