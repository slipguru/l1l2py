.. _l1l2py:

********************
L1L2Py Documentation
********************

:Release: |release|
:Date: |today|
:Homepage: http://slipguru.disi.unige.it/Software/L1L2Py
:Download: http://slipguru.disi.unige.it/Software/L1L2Py/L1L2Py-1.0.5.tar.gz

.. sectionauthor:: Salvatore Masecchia <salvatore.masecchia@unige.it>
.. sectionauthor:: Annalisa Barla <annalisa.barla@unige.it>

**L1L2Py** is a Python package to perform feature selection by means
of :math:`\ell_1\ell_2` regularization with double optimization.

L1L2Py makes use of `NumPy <http://numpy.scipy.org>`_ to provide fast
N-dimensional array manipulation. It is licensed under
`GNU General Public License (GPL) version 3 <http://www.gnu.org/licenses/gpl.html>`_.

L1L2Py is based on the minimization of the (naive) :math:`\ell_1\ell_2`
functional introduced in [Zou05]_ using the algorithm studied from the
theoretical viewpoint in [DeMol09a]_.
The current implementation exploits the minimization algorithm proposed in
[Beck09]_.

L1L2Py is the Python implementation of the one proposed and applied
in [DeMol09b]_.
It consists of two stages. The first one identifies the minimal
set of relevant variables (in terms of prediction error).
Starting from the minimal list, the second stage selects the family of
(almost completely) nested lists of relevant variables for increasing values
of linear correlation.

.. toctree::
   :maxdepth: 2

   tutorial.rst
   core.rst
   algorithms.rst
   tools.rst

:ref:`genindex`


.. rubric:: References

.. [Zou05]    H. Zou, T. Hastie,
              "Regularization and variable selection via the elastic net"
              J.R. Statist. Soc. B, 67 (2) pp. 301-320, 2005
.. [DeMol09a] C. De Mol, E. De Vito, L. Rosasco,
              "Elastic-net regularization in learning theory"
              Journal of Complexity, n. 2, vol. 25, pp. 201-230, 2009.
.. [DeMol09b] C. De Mol, S. Mosci, M. Traskine, A. Verri,
              "A Regularized Method for Selecting Nested Group of Genes from
              Microarray Data"
              Journal of Computational Biology, vol. 16, pp. 677-690, 2009.
.. [Beck09]   A. Beck, M. Teboulle,
              "A fast iterative shrinkage-thresholding algorithm for linear
              inverse problems"
              SIAM Journal on Imaging Sciences, 2(1):183–202, Mar 2009.
