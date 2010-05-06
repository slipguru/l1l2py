from algorithms import *
from algorithms import _maximum_eigenvalue, _soft_thresholding, _functional

import numpy as np

def l1l2_regularization_STD(data, labels, mu, tau, beta=None, kmax=1e5,
                            tol=1e-6, returns_iterations=False):
    n = data.shape[0]

    # Useful quantities
    sigma = _maximum_eigenvalue(data)/n + mu
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = data.T / (n * sigma)
    XTY = np.dot(XT, labels)

    if beta is None:
        beta = np.zeros_like(XTY)

    values = list()
    values.append(_functional(data, labels, beta, tau, mu))

    k, kmin = 0, 10
    th, difference = -np.inf, np.inf
    #while k < kmin or ((difference > th).any() and k < kmax):
    while k < kmin or (distance > th and k < kmax):
        k += 1

        value = beta +  XTY - np.dot(XT, np.dot(data, beta))
        beta_next = _soft_thresholding(value, tau_s) / (1.0 + mu_s)

        values.append(_functional(data, labels, beta_next, tau, mu))

        # Convergence values
        difference = np.abs(beta_next - beta)
        distance = np.linalg.norm(difference)
        th = np.linalg.norm(beta) * (tol)# / k)
        #th = np.abs(beta) * (tol / k)

        beta = beta_next

    if returns_iterations:
        return beta, k, values
    else:
        return beta

def l1l2_regularization_FISTA(data, labels, mu, tau, beta=None, kmax=1e5,
                              tolerance=1e-6, returns_iterations=False):
    n, d = data.shape

    # Useful quantities
    sigma = _maximum_eigenvalue(data)/n + mu
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = data.T / (n * sigma)
    XTY = np.dot(XT, labels)

    # beta starts from 0 and we assume also that the previous value is 0
    if beta is None:
        beta = np.zeros_like(XTY)
    beta_prev = beta

    # Auxiliary beta (FISTA implementation), starts from 0
    aux_beta = beta
    t, t_next = 1, None     # t values initializations

    values = list()
    values.append(_functional(data, labels, beta_prev, tau, mu))

    k, kmin = 0, 10
    th, difference = -np.inf, np.inf
    while k < kmin or (distance > th and k < kmax):
        k += 1

        # New solution
        value = (1.0 - mu_s)*aux_beta + XTY - np.dot(XT, np.dot(data, aux_beta))
        beta = _soft_thresholding(value, tau_s)

        values.append(_functional(data, labels, beta, tau, mu))

        # New auxiliary beta (FISTA)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t*t))
        difference = (beta - beta_prev)
        aux_beta = beta + ((t - 1.0)/t_next)*difference

        # Convergence values
        distance = np.linalg.norm(difference)
        th = np.linalg.norm(beta) * tolerance

        # Values update
        beta_prev, t = beta, t_next

    if returns_iterations:
        return beta, k, values
    else:
        return beta
