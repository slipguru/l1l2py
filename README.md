<p align="center">
  <img src="http://www.slipguru.unige.it/Software/L1L2Py/_static/l1l2py_logo.png"><br><br>
</p>
-----------------
![travis-status](https://travis-ci.org/slipguru/l1l2py.svg?branch=master)

**l1l2py** is a Python package to perform variable selection by means
of l1l2 regularization with double optimization.

**l1l2py** makes use of `numpy` <http://numpy.scipy.org> to provide fast
N-dimensional array manipulation and is the Python implementation of the
method proposed and applied in [DeMol09]_.

Also, it is fully compatible with `scikit-learn` Python toolbox for machine learning <http://scikit-learn.org/stable/>.

**l1l2py** is a project of SLIPGURU - Statistical Learning and Image Processing
Genoa University Research Group - Via Dodecaneso, 35 - 16146 Genova, ITALY
<http://slipguru.unige.it/>.

**l1l2py** is free software. It is licensed under the GNU General Public
License (GPL) version 3 <http://www.gnu.org/licenses/gpl.html>.

## Installation

**l1l2py** supports Python 2.7

### Pip installation
`$ pip install L1L2py`

### Installing from sources
```bash
$ git clone https://github.com/slipguru/l1l2py
$ cd l1l2py
$ python setup.py install
```

## Try L1L2py
**l1l2py** can be used to solve both regression and classification problems.

### Regression
```python
>>> from l1l2py.linear_model import L1L2
>>> from sklearn.datasets import load_boston
>>> X, y = load_boston(return_X_y=True)
>>> l1l2 = L1L2(tau=1, mu=0.5).fit(X, y)
>>> l1l2
L1L2(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5, max_iter=10000,
   mu=0.5, normalize=False, positive=False, precompute=False,
   random_state=None, selection='cyclic', tau=1, tol=0.0001, use_gpu=False,
   warm_start=False)
>>> l1l2.coef_
array([-0.07871197,  0.05147779, -0.00579328,  0.        , -0.        ,
        1.03468713,  0.01992332, -0.74193449,  0.30231875, -0.01640304,
       -0.77808095,  0.00851397, -0.75325201])
```

### Classification
```python
>>> from l1l2py.classification import L1L2Classifier
>>> from sklearn.datasets import load_breast_cancer
>>> X, y = load_breast_cancer(return_X_y=True)
>>> l1l2 = L1L2Classifier(tau=1, mu=0.5).fit(X, y)
>>> l1l2
L1L2Classifier(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
   max_iter=10000, mu=0.5, normalize=False, positive=False,
   precompute=False, random_state=None, selection='cyclic', tau=1,
   threshold=1e-16, tol=0.0001, use_gpu=False, warm_start=False)
>>> l1l2.coef_
array([[-0.        , -0.        ,  0.00013437,  0.00046001, -0.        ,
        -0.        , -0.        , -0.        , -0.        , -0.        ,
        -0.        , -0.        , -0.        ,  0.        , -0.        ,
        -0.        , -0.        , -0.        , -0.        , -0.        ,
        -0.        , -0.01317881, -0.03120279,  0.00032235, -0.        ,
        -0.        , -0.        , -0.        , -0.        , -0.        ]])
```

## More info
Check the [L1L2py documentation](http://www.slipguru.unige.it/Software/L1L2Py/).

## Reference
[DeMol09] C. De Mol, S. Mosci, M. Traskine, A. Verri,
         "A Regularized Method for Selecting Nested Group of Genes from
         Microarray Data"
        Journal of Computational Biology, vol. 16, pp. 677-690, 2009.

