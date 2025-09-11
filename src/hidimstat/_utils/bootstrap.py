import numpy as np
from sklearn.utils import resample
from hidimstat._utils.utils import check_random_state


def _subsampling(n_samples, train_size, groups=None, random_state=None):
    """
    Random subsampling for statistical inference.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    train_size : float
        Fraction of samples to include in the training set (between 0 and 1).
    groups : ndarray, shape (n_samples,), optional (default=None)
        Group labels for samples.
        If not None, a subset of groups is selected.
    random_state : int, optional (default=None)
        Random seed for reproducibility.

    Returns
    -------
    train_index : ndarray
        Indices of selected samples for training.
    """
    rng = check_random_state(random_state)
    index_row = np.arange(n_samples) if groups is None else np.unique(groups)
    train_index = resample(
        index_row,
        n_samples=int(len(index_row) * train_size),
        replace=False,
        random_state=rng,
    )
    if groups is not None:
        train_index = np.arange(n_samples)[np.isin(groups, train_index)]
    return train_index
