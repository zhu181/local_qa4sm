import numpy as np
import pandas as pd
import pytest
from qa4sm_gpu_validation.gpu_metrics import BatchedPairwiseMetrics, BatchedTCAMetrics


@pytest.fixture
def synthetic_pairwise_data():
    """Generate synthetic pairwise data for 5 grid points."""
    np.random.seed(42)
    data_list = []
    gpi_infos = []
    for i in range(5):
        n = 100
        x = np.random.randn(n) + i * 0.1
        y = x * 0.8 + np.random.randn(n) * 0.3
        df = pd.DataFrame({"ds1": x, "ds2": y})
        data_list.append(df)
        gpi_infos.append((i, i * 0.1, i * 0.2))
    return data_list, gpi_infos


def test_pairwise_metrics_basic(synthetic_pairwise_data):
    """Test that pairwise metrics are computed for all grid points."""
    data_list, gpi_infos = synthetic_pairwise_data
    calc = BatchedPairwiseMetrics(calc_kendall=False)
    results = calc.calc_batch(data_list, gpi_infos)

    assert len(results) == 5
    for r in results:
        assert "n_obs" in r
        assert "status" in r
        assert "BIAS" in r
        assert "RMSD" in r
        assert "R" in r
        assert r["status"][0] == 0  # OK status


def test_pairwise_metrics_match_numpy(synthetic_pairwise_data):
    """Test that GPU pairwise metrics match numpy computation within tolerance."""
    data_list, gpi_infos = synthetic_pairwise_data
    calc = BatchedPairwiseMetrics(calc_kendall=False)
    results = calc.calc_batch(data_list, gpi_infos)

    # Compare first point with manual numpy computation
    df = data_list[0].dropna()
    x, y = df.values[:, 0], df.values[:, 1]
    expected_bias = np.mean(x) - np.mean(y)
    expected_rmsd = np.sqrt(np.mean((x - y) ** 2))

    assert abs(results[0]["BIAS"][0] - expected_bias) < 1e-5
    assert abs(results[0]["RMSD"][0] - expected_rmsd) < 1e-5


def test_pairwise_insufficient_data():
    """Test handling of insufficient data points."""
    # DataFrame with only 5 rows (< min_obs=10)
    df = pd.DataFrame({"ds1": [1, 2, 3, 4, 5], "ds2": [2, 3, 4, 5, 6]})
    calc = BatchedPairwiseMetrics(min_obs=10)
    results = calc.calc_batch([df], [(1, 0.1, 0.2)])

    assert len(results) == 1
    assert results[0]["status"][0] == -1  # Insufficient
    assert np.isnan(results[0]["BIAS"][0])  # NaN metrics


def test_pairwise_mixed_valid_invalid():
    """Test batch with mix of valid and invalid data."""
    np.random.seed(42)
    valid_df = pd.DataFrame({"ds1": np.random.randn(100), "ds2": np.random.randn(100)})
    invalid_df = pd.DataFrame({"ds1": [1, 2], "ds2": [3, 4]})  # too short

    calc = BatchedPairwiseMetrics(min_obs=10)
    results = calc.calc_batch([valid_df, invalid_df], [(1, 0.1, 0.2), (2, 0.3, 0.4)])

    assert len(results) == 2
    assert results[0]["status"][0] == 0  # Valid
    assert results[1]["status"][0] == -1  # Invalid


@pytest.fixture
def synthetic_tca_data():
    """Generate synthetic TCA data for 3 grid points."""
    np.random.seed(42)
    data_list = []
    gpi_infos = []
    for i in range(3):
        n = 100
        truth = np.random.randn(n)
        x = truth + np.random.randn(n) * 0.1  # ref
        y = truth * 1.2 + np.random.randn(n) * 0.2
        z = truth * 0.8 + np.random.randn(n) * 0.3
        df = pd.DataFrame({"ref": x, "ds1": y, "ds2": z})
        data_list.append(df)
        gpi_infos.append((i, i * 0.1, i * 0.2))
    return data_list, gpi_infos


def test_tca_metrics_basic(synthetic_tca_data):
    """Test that TCA metrics are computed for all grid points."""
    data_list, gpi_infos = synthetic_tca_data
    calc = BatchedTCAMetrics(refname="ref", dataset_names=["ref", "ds1", "ds2"], min_obs=10)
    results = calc.calc_batch(data_list, gpi_infos)

    assert len(results) == 3
    for r in results:
        assert ("snr", "ref") in r
        assert ("snr", "ds1") in r
        assert ("snr", "ds2") in r
        assert ("err_std", "ref") in r
        assert ("beta", "ref") in r
        assert r[("beta", "ref")][0] == pytest.approx(1.0)  # ref beta is 1


def test_tca_metrics_snr_reasonable(synthetic_tca_data):
    """Test that SNR values are reasonable (not NaN, not inf)."""
    data_list, gpi_infos = synthetic_tca_data
    calc = BatchedTCAMetrics(refname="ref", dataset_names=["ref", "ds1", "ds2"], min_obs=10)
    results = calc.calc_batch(data_list, gpi_infos)

    for r in results:
        for ds in ["ref", "ds1", "ds2"]:
            snr = r[("snr", ds)][0]
            assert not np.isnan(snr)
            assert not np.isinf(snr)


def test_tca_bootstrap_cis(synthetic_tca_data):
    """Test that bootstrap CIs are computed correctly."""
    data_list, gpi_infos = synthetic_tca_data
    calc = BatchedTCAMetrics(refname="ref", dataset_names=["ref", "ds1", "ds2"], min_obs=10)
    results = calc.calc_batch_with_bootstrap(data_list, gpi_infos, nsamples=50, alpha=0.05)

    assert len(results) == 3
    for r in results:
        assert ("snr_ci_lower", "ref") in r
        assert ("snr_ci_upper", "ref") in r
        lower = r[("snr_ci_lower", "ref")][0]
        upper = r[("snr_ci_upper", "ref")][0]
        assert lower <= upper
