import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def kfold_errors(errors, ranges, labels):
    "X rows, Y columns"
    fig = plt.figure()
    ax = Axes3D(fig)
    
    x_vals, y_vals = np.meshgrid(*ranges)
    x_idxs, y_idxs = np.meshgrid(*(np.arange(len(x)) for x in ranges))
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel('$error$')

    ax.plot_surface(x_vals, y_vals, errors[x_idxs, y_idxs],
                    rstride=1, cstride=1, cmap=cm.jet)
    
        
if __name__ == "__main__":
    import tools
    errors = np.empty((20, 10))
    x = tools.geometric_range(1e-1, 1e4, 20)
    y = tools.geometric_range(1e-1, 1e4, 10)
    print x
    
    for i in xrange(20):
        for j in xrange(10):    
            errors[i, j] = (x[i] * y[j])
    
    kfold_errors(errors, (np.log10(x), np.log10(y)),
                 ('$log_{10}(x)$', '$log_{10}(y)$'))
    
    plt.show()
    
