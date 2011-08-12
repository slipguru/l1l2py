import numpy as np

def _check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

class KFold(object):
    r"""k-fold cross validation splits.

    Given a list of labels, the function produces a list of ``k`` splits.
    Each split is a pair of tuples containing the indexes of the training set
    and the indexes of the test set.

    Parameters
    ----------
    labels : array_like, shape (N,)
        Data labels.
    k : int, greater than `0`
        Number of splits.
    rseed : int, optional (default is `0`)
        Random seed.

    Returns
    -------
    splits : list of ``k`` tuples
        Each tuple contains two lists with the training set and test set
        indexes.

    Raises
    ------
    ValueError
        If ``k`` is less than 2 or greater than `N`.

    Examples
    --------
    >>> labels = range(10)
    >>> l1l2py.tools.kfold_splits(labels, 2)
    [([7, 1, 3, 6, 8], [9, 4, 0, 5, 2]), ([9, 4, 0, 5, 2], [7, 1, 3, 6, 8])]
    >>> l1l2py.tools.kfold_splits(labels, 1)
    Traceback (most recent call last):
        ...
    ValueError: 'k' must be greater than one and smaller or equal than the number of samples

    """
    def __init__(self, n, k, random_state=None):
        self.n = n
        self.k = k
        self.random_state = _check_random_state(random_state)
        
    def __iter__(self):
        if not (2 <= self.k <= self.n):
            raise ValueError("'k' must be greater than one and smaller or equal "
                             "than the number of samples")

        indexes = range(self.n)
        self.random_state.shuffle(indexes)

        for split in KFold._splits(indexes, self.k):
            yield split
    
    @staticmethod
    def _splits(indexes, k):
        """Splits the 'indexes' list in input in k disjoint chunks."""
        for start, end in KFold._split_dimensions(len(indexes), k):
            yield (indexes[:start] + indexes[end:], indexes[start:end])
                
    @staticmethod
    def _split_dimensions(num_items, num_splits):
        """Generator wich gives the pairs of indexes to split 'num_items' data
           in 'num_splits' chunks."""
        start = 0
        remaining_items = float(num_items)
    
        for remaining_splits in xrange(num_splits, 0, -1):
            split_size = int(round(remaining_items / remaining_splits))
            end = start + split_size
    
            yield start, end
    
            start = end
            remaining_items -= split_size
            
    def __len__(self):
        return self.k
    
class StratifiedKFold(KFold):
    """Sstratified k-fold cross validation splits.

    This function is a variation of ``kfold_splits``, which
    returns stratified splits. The divisions are made by preserving
    the percentage of samples for each class, assuming that the problem
    is binary.

    Parameters
    ----------
    labels : array_like, shape (N,)
        Data labels (usually contains only 1s and -1s).
    k : int, greater than `0`
        Number of splits.
    rseed : int, optional (default is `0`)
        Random seed.

    Returns
    -------
    splits : list of ``k`` tuples
        Each tuple contains two lists with the training set and test set
        indexes.

    Raises
    ------
    ValueError
        If `labels` contains more than two classes labels.
    ValueError
        If ``k`` is less than 2 or greater than number of positive or negative
        samples in `labels`.

    Examples
    --------
    >>> labels = range(10)
    >>> l1l2py.tools.stratified_kfold_splits(labels, 2)
    Traceback (most recent call last):
        ...
    ValueError: 'labels' must contains only two class labels
    >>> labels = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1]
    >>> l1l2py.tools.stratified_kfold_splits(labels, 2)
    [([8, 9, 5, 2, 1], [7, 6, 3, 0, 4]), ([7, 6, 3, 0, 4], [8, 9, 5, 2, 1])]
    >>> l1l2py.tools.stratified_kfold_splits(labels, 1)
    Traceback (most recent call last):
        ...
    ValueError: 'k' must be greater than one and smaller or equal than number of positive and negative samples

    """
    def __init__(self, labels, k, random_state=None):
        self.labels = labels
        self.k = k
        self.random_state = _check_random_state(random_state)
        
    def __iter__(self):
        classes = np.unique(self.labels)
        if classes.size != 2:
            raise ValueError("'labels' must contains only two class labels")
    
        n_indexes = (np.where(self.labels == classes[0])[0]).tolist()
        p_indexes = (np.where(self.labels == classes[1])[0]).tolist()
    
        if not (2 <= self.k <= min(len(n_indexes), len(p_indexes))):
            raise ValueError("'k' must be greater than oen and smaller or equal "
                             "than number of positive and negative samples")
    
        self.random_state.shuffle(n_indexes)
        n_splits = list(KFold._splits(n_indexes, self.k))
        
        self.random_state.shuffle(p_indexes)
        p_splits = list(KFold._splits(p_indexes, self.k))
    
        for ns, ps in zip(n_splits, p_splits):
            train = ns[0] + ps[0]
            test = ns[1] + ps[1]
            yield (train, test)
    
    
