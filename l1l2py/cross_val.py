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
    
    
