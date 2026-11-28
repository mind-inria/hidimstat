from hidimstat._utils.samplers import subsampling
from hidimstat._utils.utils import check_random_state


def test_subsample_size():
    n_samples = 50
    rng = check_random_state(42)

    samples = subsampling(
        n_samples=n_samples,
        train_size=1,
        groups=None,
        random_state=rng,
    )

    assert len(samples) == n_samples

    samples = subsampling(
        n_samples=n_samples,
        train_size=0.5,
        groups=None,
        random_state=rng,
    )

    assert len(samples) == n_samples // 2
