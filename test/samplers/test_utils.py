from hidimstat._utils.utils import check_random_state
from hidimstat.samplers.utils import _subsampling


def test_subsample_size():
    n_samples = 50
    rng = check_random_state(42)

    samples = _subsampling(
        n_samples=n_samples,
        train_size=1,
        groups=None,
        random_state=rng,
    )

    assert len(samples) == n_samples

    samples = _subsampling(
        n_samples=n_samples,
        train_size=0.5,
        groups=None,
        random_state=rng,
    )

    assert len(samples) == n_samples // 2
