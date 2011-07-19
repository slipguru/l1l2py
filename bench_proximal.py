"""
Does two benchmarks

First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.

In both cases, only 10% of the features are informative.
"""
import sys

import numpy as np
import gc
from time import time
from scikits.learn.datasets.samples_generator import make_regression_dataset

alpha = 0.1
# alpha = 0.01


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def bench(factory, X, Y, X_test, Y_test, ref_coef):
    gc.collect()

    # start time
    tstart = time()
    clf = factory(alpha=alpha).fit(X, Y)
    delta = (time() - tstart)
    # stop time

    print "duration: %0.3fs" % delta
    print "rmse: %f" % rmse(Y_test, clf.predict(X_test))
    print "mean coef abs diff: %f" % abs(ref_coef - clf.coef_.ravel()).mean()
    return delta

def main():
    from l1l2py.proximal import Lasso as ProximalLasso
    from scikits.learn.linear_model import Lasso as ScikitLasso
    # Delayed import of pylab
    import pylab as pl


    # the number of variable is fixed
    # and the number of point increases

    scikit_results = []
    proximal_results = []
    n = 20
    step = 500
    n_features = 1000
    n_informative = n_features / 10
    n_test_samples = 1000
    for i in range(1, n + 1):
        print '=================='
        print 'Iteration %s of %s' % (i, n)
        print '=================='
        X, Y, X_test, Y_test, coef = make_regression_dataset(
            n_train_samples=(i * step), n_test_samples=n_test_samples,
            n_features=n_features, noise=0.1, n_informative=n_informative)

        print "benching proximal: "
        proximal_results.append(bench(ProximalLasso, X, Y, X_test, Y_test, coef))
        print "benching scikits: "
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef))

    pl.clf()
    xx = range(0, n*step, step)
    pl.title('Lasso regression on sample dataset (%d features)' % n_features)
    pl.plot(xx, scikit_results, 'b-', label='scikit-learn')
    pl.plot(xx, proximal_results,'r-', label='proximal')
    pl.legend()
    pl.xlabel('number of samples to classify')
    pl.ylabel('time (in seconds)')
    pl.show()

    # now do a bench where the number of points is fixed
    # and the variable is the number of features

    scikit_results = []
    glmnet_results = []
    n = 20
    step = 100
    n_samples = 500

    for i in range(1, n + 1):
        print '=================='
        print 'Iteration %02d of %02d' % (i, n)
        print '=================='
        n_features = i * step
        n_informative = n_features / 10
        X, Y, X_test, Y_test, coef_ = make_regression_dataset(
            n_train_samples=n_samples, n_test_samples=n_test_samples,
            n_features=n_features, noise=0.1, n_informative=n_informative)

        print "benching proximal: "
        proximal_results.append(bench(ProximalLasso, X, Y, X_test, Y_test, coef_))
        print "benching scikits: "
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))

    xx = np.arange(100, 100 + n * step, step)
    pl.figure()
    pl.title('Regression in high dimensional spaces (%d samples)' % n_samples)
    pl.plot(xx, scikit_results, 'b-', label='scikit-learn')
    pl.plot(xx, proximal_results,'r-', label='proximal')
    pl.legend()
    pl.xlabel('number of features')
    pl.ylabel('time (in seconds)')
    pl.axis('tight')
    pl.show()


if __name__ == '__main__':
    import pylab as pl
    import l1l2py.algorithms
    from l1l2py.algorithms import l1l2_regularization
    from l1l2py.proximal import Lasso

    from scikits.learn.linear_model.cd_fast import enet_coordinate_descent
    from scikits.learn.linear_model import Lasso as LassoSKL

    X, Y, X_test, Y_test, coef = make_regression_dataset(
            n_train_samples=100, n_test_samples=50,
            n_features=100000, noise=0.1, n_informative=100)

    alpha = 2.
    tau = 2.*alpha
    mu = 0.0

    def _functional(beta):
        n = X.shape[0]

        Xc, Yc, Xmean, Ymean = Lasso._center_data(X, Y, True)

        loss = Yc.ravel() - np.dot(Xc, beta.ravel())
        loss_quadratic_norm = np.linalg.norm(loss) ** 2
        beta_quadratic_norm = np.linalg.norm(beta) ** 2
        beta_l1_norm = np.abs(beta).sum()

        return (((1./n) * loss_quadratic_norm)
                 + mu * beta_quadratic_norm
                 + tau * beta_l1_norm)

    import time

    st = time.time()
    clfcd = LassoSKL(alpha=alpha, max_iter=100000).fit(X, Y)
    cdt = time.time() - st

    st = time.time()
    clfprox = Lasso(alpha=alpha, adaptive=True).fit(X, Y)
    proxt = time.time() - st
    
    print
    print 'Calculated Coefficients'
    #realcoef = coef[coef.nonzero()]
    proxcoef = clfprox.coef_[clfprox.coef_.nonzero()]
    cdcoef = clfcd.coef_[clfcd.coef_.nonzero()]
    #print 'Real:', realcoef
    print 'Prox:', proxcoef
    print 'Cd:  ', cdcoef
    print 'Max Diff:', np.abs(proxcoef - cdcoef).max()
    
    #print 'Max Differences'
    #print 'Prox:', np.abs(proxcoef - realcoef).max()
    #print 'Cd:  ', np.abs(proxcoef - realcoef).max()
    
    print
    print 'Prox: %d iter in %.3f sec' % (clfprox.niter_, proxt)
    print 'Cd: convergence in %.3f sec' % cdt
    
    print
    valueprox = _functional(clfprox.coef_)
    valuecd = _functional(clfcd.coef_)
    print 'Minimum reached by prox:', valueprox
    print 'Minimum reached by cd:', valuecd
    print 'Diff:', np.abs(valueprox - valuecd)
    
    # Proximal perfomances
    #pl.plot(clfprox.energy_, 'r', label='prox')
    #pl.axhline(min(clfprox.energy_), c='r', ls='--')
    #pl.axhline(valueprox, c='r', ls='-.')
    #pl.axhline(valuecd, c='b', ls='--')
    #
    #pl.legend()
    #pl.title('func vs iter')
    #pl.show()


# lprun
#import numpy as np
#from scikits.learn.datasets.samples_generator import make_regression_dataset
#import l1l2py.algorithms
#from l1l2py.algorithms import l1l2_regularization
#from l1l2py.proximal import Lasso
#
#X, Y, X_test, Y_test, coef = make_regression_dataset(
#        n_train_samples=100, n_test_samples=50,
#        n_features=10000, noise=0.1, n_informative=100)
#
#alpha = 2.
#tau = 2.*alpha
#mu = 0.0
#
#%lprun -f l1l2_regularization Lasso(alpha=alpha, adaptive=True).fit(X, Y)