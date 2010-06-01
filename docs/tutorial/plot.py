## This code is written by Salvatore Masecchia <salvatore.masecchia@unige.it>
## and Annalisa Barla <annalisa.barla@unige.it>
## Copyright (C) 2010 SlipGURU -
## Statistical Learning and Image Processing Genoa University Research Group
## Via Dodecaneso, 35 - 16146 Genova, ITALY.
##
## This file is part of L1L2Py.
##
## L1L2Py is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## L1L2Py is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with L1L2Py. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def kcv_errors(errors, range_x, range_y, label_x, label_y):
    r"""Plot a 3D error surface.

    Parameters
    ----------
    errors : (N, D) ndarray
        Error matrix.
    range_x : array_like of N values
        First axis values.
    range_y : array_like of D values
        Second axis values.
    label_x : str
        First axis label.
    label_y : str
        Second axis label.

    Examples
    --------
    >>> errors = numpy.empty((20, 10))
    >>> x = numpy.arange(20)
    >>> y = numpy.arange(10)
    >>> for i in range(20):
    ...     for j in range(10):
    ...         errors[i, j] = (x[i] * y[j])
    ...
    >>> kcv_errors(errors, x, y, 'x', 'y')
    >>> plt.show()

    """
    fig = plt.figure()
    ax = Axes3D(fig)

    x_vals, y_vals = np.meshgrid(range_x, range_y)
    x_idxs, y_idxs = np.meshgrid(np.arange(len(range_x)),
                                 np.arange(len(range_y)))

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel('$error$')

    ax.plot_surface(x_vals, y_vals, errors[x_idxs, y_idxs],
                    rstride=1, cstride=1, cmap=cm.jet)
