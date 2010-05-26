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
       
