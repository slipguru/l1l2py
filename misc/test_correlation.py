#-*- coding: utf-8 -*-
import itertools as it
import time

import numpy as np
import pylab as plt

from scikits.learn.linear_model import LinearRegression, Lasso as Lasso_glmnet
from scikits.learn.linear_model import LassoLARS

from l1l2py.data import correlated_dataset
from l1l2py.proximal import Lasso
from l1l2py.algorithms import l1_bound

# ALGORITMS PARAMETERS SELECTION ##############################################
ALGORITHM = 'glmnet'
#ALGORITHM = 'proximal'

TOL = 1e-7 # a standard tolerance (prox)  err ~1e-2/3 (std s.r.)
#TOL = 1e-8 # a standard tolerance (glm) very close to zero (std s.r.)
MAX_ITER = int(1e6)

#TOL = 1e-4 # Proximal con tolleranza bassa seleziona tutte le variabili
#           # correlate... si può usare l'early stopping come criterio
#           # in qualche contesto per i metodi prossimali?
#           # glmnet non ha questa proprieta'
# La step size adattiva sempbra uccidere le prestazioni con
# dataset molto correlati... provare
# Maggiore è la correlazione tra alcune variabili, più piccola deve essere
# la tolleranza
# Comunque il numero di 1000 iterazioni di glmnet è davvero molto basso
# Come faccio a regolare in automatico la tolleranza in base
# alla correlazione della matrice?

if ALGORITHM == 'glmnet':
    LassoSolver = Lasso_glmnet
    parameters = lambda x: {'alpha': x/2.0, 'tol': TOL, 'max_iter': MAX_ITER}
    convergence = lambda clf: str(clf.dual_gap_ <= clf.eps_)
elif ALGORITHM == 'proximal':
    LassoSolver = Lasso
    parameters = lambda x: {'tau': x, 'tol': TOL, 'max_iter': MAX_ITER,
                            'adaptive_step_size': False}
    convergence = lambda clf: str(clf.niter_ < clf.max_iter-1)
###############################################################################

# MATRIX PARAMETERS ###########################################################
SIMPLE_LASSO = {
    'num_samples': 100,
    'num_variables': 50,
    'groups_cardinality': [1, 1, 1, 1],
    'weights': [1., -2., 1.5, -2.5],
    'variables_stdev': 1.0,
    'correlations_stdev': 1e-10, # unrelevant
    'labels_stdev': 1e-2}

HIGH_DIMENSION = dict(SIMPLE_LASSO)
HIGH_DIMENSION.update({'num_samples': 30})

LESS_CORRELATED = dict(SIMPLE_LASSO)
LESS_CORRELATED.update({
    'groups_cardinality': [5, 5, 5, 5],
    'weights': [-1.]*5  + [-2.]*5  + [-1.5]*5  + [-2.5]*5,
    'correlations_stdev': 1e-1
})

HIGHLY_CORRELATED = dict(LESS_CORRELATED)
HIGHLY_CORRELATED.update({
    'correlations_stdev': 1e-3
})

VERY_HIGHLY_CORRELATED = dict(HIGHLY_CORRELATED)
VERY_HIGHLY_CORRELATED.update({
    'correlations_stdev': 1e-5
})

PARAMETERS = VERY_HIGHLY_CORRELATED
TEST_PARAMETERS = dict(PARAMETERS)
TEST_PARAMETERS['num_samples'] /= 2

# Generation ##################################################################
print """
******************************
********* PARAMETERS *********
******************************"""
for k in PARAMETERS:
    print "** %s: %s" % (k, PARAMETERS[k])
print "******************************"
print "Note: test num_samples is %s" % TEST_PARAMETERS['num_samples']
print

# Pre-saved
#file = np.load('data/VERY_HIGHLY_CORRELATED.npz')
#X, y, T, yt = [file[k] for k in ('X', 'y', 'T', 'yt')]

# New
X, y = correlated_dataset(**PARAMETERS)
T, yt = correlated_dataset(**TEST_PARAMETERS)

# Correlation matrix plot
corrX = np.corrcoef(X.T)
corrT = np.corrcoef(T.T)

plt.figure(num=1)
plt.imshow(corrX, interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Variables correlation matrix (X)')

plt.figure(num=2)
plt.imshow(corrT, interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Variables correlation matrix (T)')

# Saving
#np.savez('data/HIGHLY_CORRELATED', X=X, y=y, T=T, yt=yt)

# True model visualization ###################################################
num_relevant = sum(PARAMETERS['groups_cardinality'])
true_model = np.zeros(X.shape[1])
true_model[:num_relevant] = PARAMETERS['weights']
var_indexes = np.arange(1, len(true_model)+1)

plt.figure(num=3)
plt.title('Selection Results')
plt.plot(var_indexes, true_model, 'g*', label='model')

prev_start = 0
colors = it.cycle('bgrcmy')
group_start = np.cumsum(PARAMETERS['groups_cardinality'])
for c, gs in it.izip(colors, group_start):
    plt.vlines(var_indexes[prev_start:gs], [0.0],
               true_model[prev_start:gs], color=c)
    prev_start = gs
plt.axhline(0.0)

# OLS solution visualization #################################################
ols = LinearRegression().fit(X, y)
plt.plot(var_indexes, ols.coef_, 'r.', label='ols')

# Lasso solution visualization ###############################################
bound = l1_bound(X, y)
tau_range = np.logspace(np.log10(bound*1e-3), np.log10(bound*0.7), 20)[::-1]

# range in reversed order
errors = list()
coefs_lasso = list()
coefs_ds = list()

coefs_lars = list()
coefs_lasso_error = list()

lasso_iterations = list()

prev_coef = np.zeros(X.shape[1])
for tau in tau_range:

    # lasso
    lasso = LassoSolver(**parameters(tau))
    seconds = -time.time()
    lasso.fit(X, y, coef_init=prev_coef)
    seconds += time.time()
    prev_coef = lasso.coef_.copy()
    coefs_lasso.append(lasso.coef_)
    lasso_iterations.append(lasso.niter_)

    # Lars lasso
    lars = LassoLARS(alpha=tau/2.0, normalize=False).fit(X, y)
    approx_err = np.linalg.norm(lasso.coef_ - lars.coef_, 2) / np.linalg.norm(lars.coef_, 2)
    coefs_lasso_error.append(approx_err)
    coefs_lars.append(lars.coef_)

    # selected
    selected = np.flatnonzero(np.abs(lasso.coef_) >= 1e-6) # close to zero

    if len(selected):
        # ols
        ols = LinearRegression().fit(X[:, selected], y)
        tmp_coef = np.zeros(X.shape[1])
        tmp_coef[selected] = ols.coef_
        coefs_ds.append(tmp_coef)

        # prediction
        pred = ols.predict(T[:, selected])
        errors.append(((pred - yt)**2).mean())
    else:
        errors.append(((np.zeros_like(yt) - yt)**2).mean())

    print 'tau: %.3e; mse: %.3e; convergence: %s' % (tau, errors[-1],
                                                     convergence(lasso))
    print 'iter: %5d; seconds: %.3fs; ' \
          'max delta coef: %.3e; max coef: %.3e' % (lasso.niter_,
                                                    seconds,
                                                    lasso.delta_max_,
                                                    lasso.max_coef_)
    #print lasso.delta_max_ * lasso.max_coef_
    #print lasso.coef_
    #print lars.coef_
    print

best_lasso_coef = coefs_lasso[np.argmin(errors)]
best_ds_coef = coefs_ds[np.argmin(errors)]
best_lars_coef = coefs_lars[np.argmin(errors)]
plt.plot(var_indexes, best_lasso_coef, 'b^', label='best lasso')
plt.plot(var_indexes, best_ds_coef, 'c^', label='best unbiased')
plt.plot(var_indexes, best_lars_coef, 'y^', label='best lars')

print
print 'Best lasso model coefficients:'
print best_lasso_coef

print
print 'Best lars lasso model coefficients:'
print best_lars_coef

# Unbiased mean for each group
prev_start = 0
colors = it.cycle('bgrcmy')
group_start = np.cumsum(PARAMETERS['groups_cardinality'])
for c, gs in it.izip(colors, group_start):
    plt.hlines([best_ds_coef[prev_start:gs].mean()],
               var_indexes[prev_start], var_indexes[gs-1],
               color=c, linestyle='dashed', lw=2, alpha=0.5)
    prev_start = gs

plt.legend(loc='best')

# Lasso coefficients path visualization ######################################
plt.figure(num=4)

plt.subplot(2, 2, 1)
coefs_paths = np.asarray(coefs_lasso)
for j in range(X.shape[1]):
    plt.plot(coefs_paths[:,j], '.-')
plt.axvline(np.argmin(errors), color='r')
plt.axhline(0.0)
plt.title('Coefficients path (big to small tau)')

plt.subplot(2, 2, 2)
coefs_paths = np.asarray(coefs_lars)
for j in range(X.shape[1]):
    plt.plot(coefs_paths[:,j], '.-')
plt.axvline(np.argmin(errors), color='r')
plt.axhline(0.0)
plt.title('LARS Lasso coefficients path (big to small tau)')

plt.subplot(2, 2, 3)
plt.plot(errors)
plt.axvline(np.argmin(errors), color='r')
plt.title('Test MSE')

#plt.subplot(2, 2, 4)
#plt.plot(coefs_lasso_error)
#plt.axvline(np.argmin(errors), color='r')
#plt.title('Approximation Error')

plt.subplot(2, 2, 4)
plt.plot(lasso_iterations)
plt.axvline(np.argmin(errors), color='r')
plt.title('Iteration performed')

## Test approximation by number of iteration ##################################
#best_tau = tau_range[np.argmin(errors)]
#print
#print 'Best tau: %.3e' % best_tau
#
#max_iter_range = [int(x) for x in np.logspace(np.log10(1e5), np.log10(MAX_ITER), 10)]
#params = parameters(best_tau)
##params['tol'] = 1e-20 # very small tolerance
#
#coeffs_lasso = list()
#errors = list()
#
#prev_coef = np.zeros(X.shape[1])
#params['max_iter'] = 0
#for max_iter in max_iter_range:
#
#    # number of iteration (we start from the previous solution)
#    params['max_iter'] = max_iter# - params['max_iter']
#
#    # lasso
#    lasso = LassoSolver(**params)
#    lasso.fit(X, y)#, coef_init=prev_coef)
#    prev_coef = lasso.coef_.copy()
#    coefs_lasso.append(lasso.coef_)
#
#    #print lasso.niter_
#    print lasso.dual_gap_, lasso.eps_
#
#    # Valutation
#    approx_err = np.linalg.norm(lasso.coef_ - best_lars_coef) / np.linalg.norm(best_lars_coef)
#    errors.append(approx_err)
#
#    print 'max iter: %8d; error: %.3e; convergence: %s' % (max_iter, errors[-1],
#                                                           convergence(lasso))

# Perché glmnet mi dice sempre che converge con
# qualsiasi numero di iterazione massima ma riesce sempre a migliorare
# il risultato? Perchè si basa sul gap e non sul beta
# Il nostro fa sempre lo stesso numero di iterazioni,
# che probabilmente è lo stesso fatto durante la prima fase,
# questo perché all'epoca non era probabilmente finito a fondo scala.
# Ma sotto  il limite minimo di max iter che sto usando.
# Perché glmnet scende lo stesso solo variando le iterazioni? Le ha fatte tutte
# prima? Potrebbe essere... e poi dice che converge guardando il duality gap,
# non guardando che è arrivato al numero massimo.
# Cioè dice: potevo scendere di più se avessi avuto più iterazioni,
# però comunque il risultato è buono perché il duality gap è sotto eps...
# che a me però sembra altino... che vuol dire??
# Però se aumento solo la tolleranza, glmnet dovrebbe darmi lo stesso risultato
# se arriva a max_iter, al massimo mi dice che non converge.
# NON E' cosi'... devo rimettere le mani in glmnet per leggere il numero di
# iterazioni che fa

#plt.figure(num=5)
#plt.loglog(max_iter_range, errors)

plt.show()
