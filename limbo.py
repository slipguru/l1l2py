import numpy as np

# SFISTA (Stable FISTA?) :-D
def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=1e5,
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

    # Auxiliary beta (FISTA implementation), starts from beta
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

        # Monotonicity check
        if value_temp <= value_prev:
            beta = beta_temp
            value = value_temp
        else:
            # step-size (sigma) ??
            value = (1.0 - mu_s)*beta_prev + XTY - np.dot(XT, np.dot(data, beta_prev))
            beta = _soft_thresholding(value, tau_s)
            value = _functional(data, labels, beta_temp, tau, mu)

        # FISTA step
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t*t))            
        difference = (beta - beta_prev)
        aux_beta = beta + ((t - 1.0)/t_next)*difference

        difference = np.abs(difference)
        th = np.abs(beta) * (tolerance / math.sqrt(k))
        
        # Values update
        beta_prev = beta
        value_prev = value
        t = t_next

    if returns_iterations:
        return beta, k
    else:
        return beta
    
def _sigma(matrix, mu):
    n, d = matrix.shape

    if d > n:
        tmp = np.dot(matrix, matrix.T)
    else:
        tmp = np.dot(matrix.T, matrix)

    return (np.linalg.eigvalsh(tmp).max()/n) + mu
       
def _functional(X, Y, beta, tau, mu):
    n = X.shape[0]

    loss = Y - np.dot(X, beta)
    loss_quadratic_norm = (loss * loss).sum()
    beta_quadratic_norm = (beta * beta).sum()
    beta_l1_norm = np.abs(beta).sum()

    return (loss_quadratic_norm/n + mu  * beta_quadratic_norm
                                  + tau * beta_l1_norm)
