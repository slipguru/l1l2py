from algorithms import *
from algorithms import _soft_thresholding
from limbo import _functional, _sigma

import numpy as np
import math
 
    
def l1l2_regularization_FISTA(data, labels, mu, tau, beta=None, kmax=1e5,
                              tolerance=1e-5, returns_iterations=False):
    n, d = data.shape

    # Useful quantities
    sigma = _sigma(data, mu)
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
    t, t_next = 1., None     # t values initializations

    k, kmin = 0, 10
    th, difference = -np.inf, np.inf
    while k < kmin or ((difference > th).any() and k < kmax):
        k += 1

        # New solution
        value = (1.0 - mu_s)*aux_beta + XTY - np.dot(XT, np.dot(data, aux_beta))
        beta = _soft_thresholding(value, tau_s)

        # New auxiliary beta (FISTA)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t*t))
        difference = (beta - beta_prev)
        aux_beta = beta + ((t - 1.0)/t_next)*difference

        # Convergence values
        difference = np.abs(difference)
        th = np.abs(beta) * (tolerance / math.sqrt(k))
        
        # Values update
        beta_prev, t = beta, t_next

    if returns_iterations:
        return beta, k, values
    else:
        return beta
    
    
def l1l2_regularization_MFISTA(data, labels, mu, tau, beta=None, kmax=1e5,
                               tolerance=1e-5, returns_iterations=False):
    n = data.shape[0]

    # Useful quantities
    sigma = _sigma(data, mu)
    mu_s = mu / sigma
    tau_s = tau / sigma
    XT = data.T / (n * sigma)
    XTY = np.dot(XT, labels)

    # beta starts from 0 and we assume the previous value is also 0
    if beta is None:
        beta = np.zeros_like(XTY)
    beta_prev = beta
    value_prev = _functional(data, labels, beta_prev, tau, mu)

    # Auxiliary beta (FISTA implementation), starts from 0
    aux_beta = beta
    t, t_next = 1., None     # t values initializations

    k, kmin = 0, 10
    th, difference = -np.inf, np.inf
    while k < kmin or ((difference > th).any() and k < kmax):
        k += 1

        # New solution
        value = (1.0 - mu_s)*aux_beta + XTY - np.dot(XT, np.dot(data, aux_beta))
        beta_temp = _soft_thresholding(value, tau_s)
        value_temp = _functional(data, labels, beta_temp, tau, mu)

        # (M)FISTA step
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t*t))
        difference = (beta_temp - beta_prev)

        # MFISTA monotonicity check
        if value_temp <= value_prev:
            beta = beta_temp
            value = value_temp
            
            aux_beta = beta + ((t - 1.0)/t_next)*difference
            
        else:          
            beta = beta_prev
            value = value_prev
            
            aux_beta = beta + (t/t_next)*difference
            
        difference = np.abs(difference)
        th = np.abs(beta) * (tolerance / math.sqrt(k))
              
        # Values update
        beta_prev = beta
        value_prev = value
        t = t_next

    if returns_iterations:
        return beta, k, values
    else:
        return beta
