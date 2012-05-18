.. _api:

==========
L1L2Py API
==========
L1L2Py models relies ont he optimization (usually minimization)
of a functional like

.. math::
    \argmin_{\boldsymbol{\beta}}
            \{f(\mathbf{X}, \boldsymbol{\beta}, \mathbf{y}) + 
              g(\boldsymbol{\beta})\},
              
where:

* :math:`\mathbf{X}` is a :math:`n \times d` data matrix
* :math:`\mathbf{y}` contains :math:`n` data labels (or regression value)
* :math:`\boldsymbol{\beta}` contains :math:`d` model coefficients
* :math:`f` is a prediction error function
* :math:`g` is a penalty term (e.g. a combination of :math:`\ell_1`
  and :math:`\ell_2` norm on :math:`\mathbf{\beta}` )
  

L1L2Py provides an abstract ``BaseModel`` that must be inherithed by all
implemented Model that match with the theoretical base functional.

**TODO**

* Model(parameters....) # Read-only (properties)
* train(X, y)
* y = predict(X)
* m.beta (property...)

