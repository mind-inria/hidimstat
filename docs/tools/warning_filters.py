import warnings

from sklearn.exceptions import ConvergenceWarning


def _reset_warning_filters(_gallery_conf, _fname):
    """Suppress sklearn FutureWarnings before each gallery example.

    TODO: remove when minimum sklearn is bumped to >= 1.7.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
