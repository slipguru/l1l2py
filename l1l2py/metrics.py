import numpy as np

from .data import center

def classification_error(labels, predictions):
    r"""Evaluate the binary classification error.

    The classification error is based on the sign of the ``predictions`` values,
    with respect to the sign of the data ``labels``.

    The function assumes that ``labels`` contains positive values for one
    class and negative values for the other one.

    .. warning::

        For efficiency reasons, the values in ``labels`` are not checked by the function.

    Parameters
    ----------
    labels : array_like, shape (N,)
        Classification labels (usually contains only 1s and -1s).
    predictions : array_like, shape (N,)
        Classification labels predicted.

    Returns
    -------
    error : float
        Classification error evaluated.

    Examples
    --------
    >>> l1l2py.tools.classification_error(labels=[1, 1, 1], predictions=[1, 1, 1])
    0.0
    >>> l1l2py.tools.classification_error(labels=[1, 1, 1], predictions=[1, 1, -1])
    0.33333333333333331
    >>> l1l2py.tools.classification_error(labels=[1, 1, 1], predictions=[1, -1, -1])
    0.66666666666666663
    >>> l1l2py.tools.classification_error(labels=[1, 1, 1], predictions=[-1, -1, -1])
    1.0
    >>> l1l2py.tools.classification_error(labels=[1, 1, 1], predictions=[10, -2, -3])
    0.66666666666666663

    """
    labels = np.asarray(labels).ravel()
    predictions = np.asarray(predictions).ravel()
    
    difference = (np.sign(labels) != np.sign(predictions))  
    return len(*difference.nonzero()) / float(len(labels))

def balanced_classification_error(labels, predictions, error_weights=None):
    r"""Returns the binary classification error balanced
    across the size of classes.

    This function returns a balanced classification error.
    With the default value for ``error_weights``, the function
    assigns greater weight to the errors belonging to the smaller class.

    Parameters
    ----------
    labels : array_like, shape (N,)
        Classification labels (usually contains only 1s and -1s).
    predictions : array_like, shape (N,)
        Classification labels predicted.
    error_weights : array_line, shape (N,), optional (default is None)
        Classification error weigths. If `None` the default weights are calculated
        removing from each value in ``labels`` their mean value.

    Returns
    -------
    error : float
        Classification error calculated.

    Examples
    --------
    >>> l1l2py.tools.balanced_classification_error(labels=[1, 1, 1], predictions=[-1, -1, -1])
    0.0
    >>> l1l2py.tools.balanced_classification_error(labels=[-1, 1, 1], predictions=[-1, 1, 1])
    0.0
    >>> l1l2py.tools.balanced_classification_error(labels=[-1, 1, 1], predictions=[1, -1, -1])
    0.88888888888888895
    >>> l1l2py.tools.balanced_classification_error(labels=[-1, 1, 1], predictions=[1, 1, 1])
    0.44444444444444442
    >>> l1l2py.tools.balanced_classification_error(labels=[-1, 1, 1], predictions=[-1, 1, -1])
    0.22222222222222224
    >>> l1l2py.tools.balanced_classification_error(labels=[-1, 1, 1], predictions=[-1, 1, -1],
    ...                                            error_weights=[1, 1, 1])
    0.33333333333333331

    """
    labels = np.asarray(labels).ravel()
    predictions = np.asarray(predictions).ravel()

    if error_weights is None:
        error_weights = np.abs(center(labels))

    errors = (np.sign(labels) != np.sign(predictions)) * error_weights
    return errors.sum() / float(len(labels))

def regression_error(labels, predictions):
    r"""Returns regression error.

    The regression error is the sum of the quadratic differences between the
    ``labels`` values and the ``predictions`` values, over the number of
    samples.

    Parameters
    ----------
    labels : array_like, shape (N,)
        Regression labels.
    predictions : array_like, shape (N,)
        Regression labels predicted.

    Returns
    -------
    error : float
        Regression error calculated.

    """
    labels = np.asarray(labels).ravel()
    predictions = np.asarray(predictions).ravel()

    difference = labels - predictions
    return np.dot(difference.T, difference).squeeze() / float(len(labels))