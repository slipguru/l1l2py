import numpy as np
import pylab as pl

from scikits.learn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
from scikits.learn.cross_val import KFold

# Data creation
np.random.seed(0)
coef = np.random.randn(200)
coef[10:] = 0.0 # only the top 10 features are impacting the model
X = np.random.randn(50, 200)
y = np.dot(X, coef) # without error
n_samples = X.shape[0]

cv = KFold(n_samples, 5)

##############################################################################
# Standard CV path
rho = 0.9
clf = ElasticNetCV(rho=rho, cv=cv)
clf.fit(X, y)

pl.figure()
pl.title('Standard ElasticNetCV')
pl.plot(clf.mse_path_)

# Parameters and preprocessing
alphas = clf.alphas
n_alphas = clf.n_alphas
path_params = clf._get_params()


def manual_cv_path(X, y, alphas, rhos, cv):
    coef_ = None  # init coef_
    folds = list(cv)
    mse_alphas = np.zeros((len(folds), len(alphas)))
    nonzero_coefs = np.zeros((len(folds), len(alphas)))
    for i, (train, test) in enumerate(folds):

        # path
        models_train = []
        for i_alpha, (alpha, rho) in enumerate(zip(alphas, rhos)):
            model = ElasticNet(alpha=alpha, rho=rho)
            model.fit(X[train], y[train], coef_init=coef_)
            coef_ = model.coef_.copy()
            models_train.append(model)

            nonzero = (model.coef_ > 1e-4)
            nonzero_coefs[i, i_alpha] = nonzero.sum()

            if nonzero_coefs[i, i_alpha] > 0:
                olsmodel = LinearRegression().fit(X[train][:,nonzero], y[train])
                y_ = olsmodel.predict(X[test][:,nonzero])
            else:
                y_ = np.zeros_like(y[test])

            mse_alphas[i, i_alpha] += ((y_ - y[test]) ** 2).mean()

    return mse_alphas.T, nonzero_coefs.T

##############################################################################
# Standard path computed manually for comparison
mse_path, nonzero_path = manual_cv_path(X, y, alphas, [rho]*len(alphas), cv)
pl.figure()
pl.title('Manually simulated ElasticNetCV')
mean_mse = mse_path.mean(axis=1)
pl.plot(mean_mse, label='mse')
pl.plot(nonzero_path.mean(axis=1), label='# selected variables')
pl.axvline(np.argmin(mean_mse))
pl.legend(loc='best')

##############################################################################
# Path with constant penalty on the l2 norm
##############################################################################
# Params conversion: tau*||w||_1 + mu*||w||_2^2
taus = alphas.copy()
mu = 1e-2 # maximum l2 regularization with alpha-rho

alphas = taus + mu
rhos = taus / (taus + mu)

mse_path, nonzero_path = manual_cv_path(X, y, alphas, rhos, cv)
pl.figure()
pl.title('Manually simulated ElasticNetCV with fixed l2 penalty')
mean_mse = mse_path.mean(axis=1)
pl.plot(mean_mse, label='mse')
pl.plot(nonzero_path.mean(axis=1), label='# selected variables')
pl.axvline(np.argmin(mean_mse))
pl.legend(loc='best')


pl.show()
