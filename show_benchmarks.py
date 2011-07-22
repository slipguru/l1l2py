import sys

import numpy as np
import pylab as pl
import cPickle as pickle

for tau in (2.0, 0.2, 0.02):
    
    n = 20
    step = 500
    n_features = 1000
    results = pickle.load(open('benchmarks/bigN_results_tau%.3f.pkl' % tau))
    pl.figure()
    xx = range(step, (n+1)*step, step)
    pl.title('Lasso regression on sample dataset (%d features)' % n_features)
    for name in results:
        pl.plot(xx, results[name], '-', label=name)
    pl.legend(loc='best')
    pl.xlabel('number of samples to classify')
    pl.ylabel('time (in seconds)')
    #pl.show()
    
    n = 20
    step = 1000#100
    n_samples = 500
    results = pickle.load(open('benchmarks/bigD_results_tau%.3f.pkl' % tau))
    xx = range(step, (n+1)*step, step)
    pl.figure()
    pl.title('Regression in high dimensional spaces (%d samples)' % n_samples)    
    for name in results:
        pl.plot(xx, results[name], '-', label=name)
    pl.legend(loc='best')
    pl.xlabel('number of features')
    pl.ylabel('time (in seconds)')
    pl.axis('tight')
    pl.show()
