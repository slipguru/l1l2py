"""Auxiliary class for range."""
import numpy as np
from collections import Sequence


class RangesScaler(object):
    """Given data and labels helps to scale L1L2 parameters ranges properly.

    This class works on tau and mu ranges passed to the l1l2 selection
    framework (see also :func:`l1l2py.model_selection` and related
    function for details).

    Scaling ranges permits to use relative (and not absolute) ranges of
    parameters.

    Attributes
    ----------
    norm_data : :class:`numpy.ndarray`
        Normalized data matrix.
    norm_labels : :class:`numpy.ndarray`
        Normalized labels vector.

    Example
    -------
    from sklearn.preprocessing import robust_scale
    from functools import partial
    data_normalizer = partial(robust_scale, with_centering=1, with_scaling=0)
    # or
    # data_normalizer = l1l2py.tools.center
    rs = RangesScaler(X, y, data_normalizer=data_normalizer)
    tau_range_scale = rs.tau_range(np.linspace(0.1, 3, 10))
    mu_range_scale = rs.mu_range(np.linspace(0.1, 3, 10))
    """

    def __init__(self, data, labels, data_normalizer=None,
                 labels_normalizer=None):
        """Init for RangesScaler."""
        self.norm_data = data
        self.norm_labels = labels
        self._tsf = self._msf = None

        if data_normalizer:
            self.norm_data = data_normalizer(self.norm_data)
        if labels_normalizer:
            self.norm_labels = labels_normalizer(self.norm_labels)

    def tau_range(self, trange):
        """Return a scaled tau range.

        Tau scaling factor is the maximum tau value to avoid and empty solution
        (where all variables are discarded).
        The value is estimated on the maximum correlation between data and
        labels.

        Parameters
        ----------
        trange : :class:`numpy.ndarray`
            Tau range containing relative values (expected maximum is lesser
            than 1.0 and minimum greater than 0.0).

        Returns
        -------
        tau_range : :class:`numpy.ndarray`
            Scaled tau range.
        """
        if np.max(trange) >= 1.0 or np.min(trange) < 0.0:
            raise ValueError('Relative tau should be in [0,1)')
        if isinstance(trange, Sequence):
            trange = np.sort(trange)
        return trange * self.tau_scaling_factor

    def mu_range(self, mrange):
        """Return a scaled mu range.

        Mu scaling factor is estimated on the maximum eigenvalue of the
        correlation matrix and is used to simplify the parameters choice.

        Parameters
        ----------
        mrange : :class:`numpy.ndarray`
            Mu range containing relative values (expected maximum is lesser
            than 1.0 and minimum greater than 0.0).

        Returns
        -------
        mu_range : :class:`numpy.ndarray`
            Scaled mu range.
        """
        if np.min(mrange) < 0.0:
            raise ValueError('Relative mu should be greater than / equal to 0')

        if isinstance(mrange, Sequence):
            mrange = np.sort(mrange)
        return mrange * self.mu_scaling_factor

    @property
    def tau_scaling_factor(self):
        """Tau scaling factor calculated on given data and labels."""
        if self._tsf is None:
            self._tsf = self._tau_scaling_factor()
        return self._tsf

    @property
    def mu_scaling_factor(self):
        """Mu scaling factor calculated on given data matrix."""
        if self._msf is None:
            self._msf = self._mu_scaling_factor()
        return self._msf

    def _tau_scaling_factor(self):
        # return l1l2py.algorithms.l1_bound(self.norm_data, self.norm_labels)
        r"""Estimation of an useful maximum bound for the `l1` penalty term.

        For each value of ``tau`` smaller than the maximum bound the solution
        vector contains at least one non zero element.

        .. warning

            That is, bounds are right if you run the `l1l2` regularization
            algorithm with the same data matrices.

        Parameters
        ----------
        data : (N, P) ndarray
            Data matrix.
        labels : (N,)  or (N, 1) ndarray
            Labels vector.

        Returns
        -------
        tau_max : float
            Maximum ``tau``.
        """
        data = self.norm_data
        labels = self.norm_labels
        corr = np.abs(np.dot(data.T, labels))
        tau_max = (corr.max() * (2.0 / data.shape[0]))
        return tau_max

    def _mu_scaling_factor(self):
        n, d = self.norm_data.shape

        if d > n:
            tmp = np.dot(self.norm_data, self.norm_data.T)
            num = np.linalg.eigvalsh(tmp).max()
        else:
            tmp = np.dot(self.norm_data.T, self.norm_data)
            evals = np.linalg.eigvalsh(tmp)
            num = evals.max() + evals.min()

        return (num / (2. * n))
