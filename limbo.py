import numpy as np

from algorithms import _maximum_eigenvalue, ridge_regression

def correlated_dataset(samples, groups, variables, true_model):
    """
    >>> correlated_dataset(30, (5, 5, 5), 40, [3.0]*15 + [0.0]*(40-15))
    ...
    """
    #print len(true_model), variables
    assert sum(groups) <= variables
    assert len(true_model) == variables

    X = np.zeros((samples, variables))
    i = 0
    for g in groups:
        variable = np.random.normal(scale=1.0, size=(samples,))
        for j in xrange(g):
            error = np.random.normal(scale=0.01, size=(samples,))
            X[:,i] = variable + error
            i+=1

    noisy = variables - sum(groups)
    X[:, i:] = np.random.normal(scale=1.0, size=(samples, noisy))

    error = np.random.normal(scale=1.0, size=(samples,))
    Y = np.dot(X, true_model) + error

    return X, Y


def l1l2_regularization(data, labels, mu, tau, beta=None, kmax=1e5,
                        tolerance=1e-5, returns_iterations=False):
    r"""Implementation of Regularized Least Squares with
    :math:`\ell_1\ell_2` penalty.

    Finds the :math:`\ell_1\ell_2` model with ``mu`` parameter associated with
    its :math:`\ell_2` norm and ``tau`` parameter associated with its
    :math:`\ell_1` norm (see `Notes`).

    Parameters
    ----------
    data : (N, D) ndarray
        Data matrix.
    labels : (N,) or (N, 1) ndarray
        Labels vector.
    mu : float
        :math:`\ell_2` norm penalty.
    tau : float
        :math:`\ell_1` norm penalty.
    beta : (D,) or (D, 1) ndarray, optional (default is `None`)
        Starting value of the iterations (see `Notes`).
        If `None`, then iterations starts from the empty model.
    kmax : int, optional (default is :math:`10^5`)
        Maximum number of iterations.
    tolerance : float, optional (default is :math:`10^{-6}`)
        Convergence tolerance (see `Notes`).
    returns_iterations : bool, optional (default is `False`)
        If `True`, returns the number of iterations performed.
        The algorithm has a predefined minimum number of iterations
        equal to `10`.

    Returns
    -------
    beta : (D,) or (D, 1) ndarray
        :math:`\ell_1\ell_2` model.
    k : int, optional
        Number of iterations performed.

    See Also
    --------
    l1l2_path

    Notes
    -----
    :math:`\ell_1\ell_2` minimizes the following objective function:

    .. math::

        \frac{1}{N} \| Y - X\beta \|_2^2 + \mu \|\beta\|_2^2 + \tau \|\beta\|_1

    finding the optimal model :math:`\beta^*`, where :math:`X` is the ``data``
    matrix and :math:`Y` contains the ``labels``.

    The computation is iterative, each step updates the value of :math:`\beta`
    until the convergence is reached [DeMol08]_:

    .. math::

        \beta^{(k+1)} = \mathbf{S}_{\frac{\tau}{\sigma}} (
                            (1 - \frac{\mu}{\sigma})\beta^k +
                            \frac{1}{n\sigma}X^T[Y - X\beta^k]
                        )

    where, :math:`\mathbf{S}_{\gamma > 0}` is the soft-thresholding function

    .. math::

        \mathbf{S}_{\gamma}(x) = sign(x) max(0, |x| - \frac{\gamma}{2})

    Moreover, the function implements a *MFISTA* [Beck09]_ modification, wich
    increases with quadratic factor the convergence rate of the algorithm.

    The constant :math:`\sigma` is a (theorically optimal) step size wich
    depends by the data:

    .. math::

        \sigma = \frac{\|X^T X\|}{N} + \mu

    The convergence is reached when:

    .. math::

        \|\beta^k - \beta^{k-1}\| \leq \|\beta^k\| * tolerance

    but the algorithm will be stop when the maximum number of iteration
    is reached.

    Examples
    --------
    >>> X = numpy.array([[0.1, 1.1, 0.3], [0.2, 1.2, 1.6], [0.3, 1.3, -0.6]])
    >>> beta = numpy.array([0.1, 0.1, 0.0])
    >>> y = numpy.dot(X, beta)
    >>> beta_rls = biolearning.algorithms.l1l2_regularization(X, y, 0.0, 1e-5)
    >>> numpy.allclose(beta, beta_rls)
    True
    >>> biolearning.algorithms.l1l2_regularization(X, y, 0.1, 0.1)
    array([ 0.        ,  0.04482757,  0.        ])

    """
    n = data.shape[0]

    # Useful quantities
    sigma = _maximum_eigenvalue(data)/n + mu
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

    # TODO: remove
    #values = list()
    #values.append(value_prev)

    k, kmin = 0, 10
    #th, distance = -np.inf, np.inf
    #while k < kmin or (distance > th and k < kmax):
    
    # OLD STOP
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
        #distance = np.linalg.norm(np.abs(difference))

        # MFISTA monotonicity check
        if value_temp <= value_prev:
            beta = beta_temp
            value = value_temp
            
            aux_beta = beta + ((t - 1.0)/t_next)*difference
            
            #print k, 'ok'
        else:
            value = (1.0 - mu_s)*beta_prev + XTY - np.dot(XT, np.dot(data, beta_prev))
            beta = _soft_thresholding(value, tau_s)
            value = _functional(data, labels, beta_temp, tau, mu)
            
            #beta = beta_prev
            #value = value_prev
            #print k, np.abs(difference).max()

            #aux_beta = beta + (t/t_next)*difference
            difference = (beta - beta_prev)
            aux_beta = beta + ((t - 1.0)/t_next)*difference

        # TODO: remove
        #values.append(value)

        # OLD STOP
        #difference = np.abs(beta - beta_prev) #np.abs(difference) -> questo fa si che si ferma alla prima non monotonia
        difference = np.abs(difference)
        th = np.abs(beta) * (tolerance / math.sqrt(k))
        #print np.where(difference > th)[0]
        #print beta.nonzero()[0]
        #print beta[np.where(difference > th)[0]].T
        #print beta.T
        
        # Convergence threshold
        #th = np.linalg.norm(beta) * tolerance
        
        # Values update
        beta_prev = beta
        value_prev = value
        t = t_next

    if returns_iterations:
        return beta, k#, values
    else:
        return beta