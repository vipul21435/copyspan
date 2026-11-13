import numpy as np
import pytest

from copyspan.stats import bootstrap_mean, paired_difference, quantiles, wilson_interval


def test_quantiles_hand_values():
    q = quantiles([4, 1, 3, 2])
    assert q["q0"] == pytest.approx(1) and q["q100"] == pytest.approx(4)
    assert q["q50"] == pytest.approx(2.5)
    assert q["q25"] == pytest.approx(1.75)
    assert quantiles([]) == {}


def test_bootstrap_is_deterministic_under_seed():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    a = bootstrap_mean(x, seed=5)
    b = bootstrap_mean(x, seed=5)
    c = bootstrap_mean(x, seed=6)
    assert a == b
    assert a != c


def test_bootstrap_matches_normal_interval_on_large_sample():
    rng = np.random.default_rng(2)
    x = rng.normal(1.0, 2.0, size=4000)
    iv = bootstrap_mean(x, n_boot=4000, seed=0)
    se = x.std(ddof=1) / np.sqrt(x.size)
    assert iv.estimate == pytest.approx(x.mean())
    assert iv.low == pytest.approx(x.mean() - 1.96 * se, abs=0.15 * se)
    assert iv.high == pytest.approx(x.mean() + 1.96 * se, abs=0.15 * se)


def test_singleton_groups_in_sorted_order_equal_record_bootstrap():
    rng = np.random.default_rng(3)
    x = rng.random(50)
    groups = [f"{i:03d}" for i in range(50)]
    clustered, plain = bootstrap_mean(x, groups, seed=9), bootstrap_mean(x, seed=9)
    assert (clustered.estimate, clustered.low, clustered.high) == pytest.approx(
        (plain.estimate, plain.low, plain.high)
    )


def test_clustered_interval_is_wider_for_correlated_groups():
    rng = np.random.default_rng(4)
    centres = rng.normal(size=40)
    x = np.repeat(centres, 25) + rng.normal(scale=0.1, size=1000)
    groups = np.repeat(np.arange(40), 25).astype(str)
    plain = bootstrap_mean(x, seed=0)
    clustered = bootstrap_mean(x, groups, seed=0)
    assert clustered.high - clustered.low > 3 * (plain.high - plain.low)
    assert clustered.estimate == pytest.approx(plain.estimate)


def test_chunking_does_not_change_result(monkeypatch):
    x = np.arange(30, dtype=float)
    whole = bootstrap_mean(x, seed=1, n_boot=100)
    import copyspan.stats as st

    monkeypatch.setattr(st, "_CHUNK_CELLS", 30 * 7)
    chunked = bootstrap_mean(x, seed=1, n_boot=100)
    assert (whole.low, whole.high) == pytest.approx((chunked.low, chunked.high))


def test_paired_difference_constant_shift():
    a = [0.5, 1.5, 2.5, 3.5]
    b = [0.0, 1.0, 2.0, 3.0]
    iv = paired_difference(a, b, seed=0)
    assert iv.estimate == pytest.approx(0.5)
    assert iv.low == pytest.approx(0.5) and iv.high == pytest.approx(0.5)
    assert iv.to_list() == [iv.low, iv.high]


def test_errors_and_empty():
    assert bootstrap_mean([]) is None
    with pytest.raises(ValueError):
        bootstrap_mean([1.0], groups=["a", "b"])
    with pytest.raises(ValueError):
        bootstrap_mean([1.0], level=1.0)
    with pytest.raises(ValueError):
        bootstrap_mean([1.0], n_boot=0)
    with pytest.raises(ValueError):
        paired_difference([1.0], [1.0, 2.0])


def test_wilson_interval_closed_forms():
    z2 = 1.959963984540054**2
    iv = wilson_interval(0, 2000)
    assert iv.estimate == pytest.approx(0.0)
    assert iv.low == pytest.approx(0.0, abs=1e-12)
    # with no successes the upper end is z^2 / (n + z^2)
    assert iv.high == pytest.approx(z2 / (2000 + z2))
    iv = wilson_interval(5, 10)
    assert iv.low + iv.high == pytest.approx(1.0)
    iv = wilson_interval(10, 10)
    assert iv.high == pytest.approx(1.0) and iv.low == pytest.approx(10 / (10 + z2))
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    with pytest.raises(ValueError):
        wilson_interval(1, 2, level=0)
