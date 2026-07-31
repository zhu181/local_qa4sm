"""Batched GPU metric calculators compatible with pytesmo's validation framework.

Computes pytesmo-compatible pairwise and triple collocation metrics for N
grid points in a single batched pass using PyTorch (GPU) with automatic
NumPy fallback when CUDA is unavailable. Result dictionaries match the format
produced by
:class:`pytesmo.validation_framework.metric_calculators.PairwiseIntercomparisonMetrics`
and
:class:`pytesmo.validation_framework.metric_calculators.TripleCollocationMetrics`.

The :class:`GPUBatchedValidation` class wraps a pytesmo ``DataManager`` and
provides a ``calc()`` method that is a drop-in replacement for
``pytesmo.validation_framework.validation.Validation.calc()``, but computes
metrics in a single batched GPU pass instead of per-grid-point.

Thread safety: HDF5/netCDF4 reads are serialized via the module-level
:data:`qa4sm_gpu_validation.read_lock.HDF5_READ_LOCK`, which is only held
during the read call itself so subsequent GPU/NumPy compute stays parallel.
Each worker thread is expected to use its own :class:`GPUBatchedValidation`
instance (see the per-thread cache pattern in the validator package).
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pytesmo.validation_framework.data_manager import get_result_combinations
from pytesmo.validation_framework.data_scalers import DefaultScaler
from scipy import stats

from qa4sm_gpu_validation.gpu_backend import get_gpu_backend

LOGGER = logging.getLogger(__name__)

_SUFFIX = ""

_STATUS_OK = 0
_STATUS_INSUFFICIENT = -1

_ALWAYS_METRICS = (
    "BIAS",
    "RMSD",
    "urmsd",
    "mse",
    "mse_corr",
    "mse_var",
    "mse_bias",
    "RSS",
    "R",
    "p_R",
)


def _to_numpy(tensor: Any) -> np.ndarray:
    """Transfer a PyTorch tensor / NumPy array to host NumPy array."""
    return get_gpu_backend().to_numpy(tensor)


def _nanmean_torch(x: Any, mask: Any, dim: int) -> Any:
    """Mean of ``x`` ignoring masked-out (padded NaN) entries along ``dim``.

    Uses PyTorch tensor operations when given tensors, NumPy otherwise.
    Padded positions carry NaN in ``x`` and 0 in ``mask``;
    ``NaN * 0`` is NaN which ``nansum`` treats as zero.
    """
    if isinstance(x, np.ndarray):
        return np.nansum(x * mask, axis=dim) / mask.sum(axis=dim)
    backend = get_gpu_backend()
    torch = backend.torch
    if torch is None:
        return np.nansum(x * mask, axis=dim) / mask.sum(axis=dim)
    return torch.nansum(x * mask, dim=dim) / mask.sum(dim=dim)


class BatchedPairwiseMetrics:
    """Compute pairwise metrics for N grid points in a batched GPU pass."""

    def __init__(
        self,
        metadata_template: dict | None = None,
        calc_spearman: bool = True,
        calc_kendall: bool = False,
        min_obs: int = 10,
    ) -> None:
        self.metadata_template = metadata_template or {}
        self.calc_spearman = calc_spearman
        self.calc_kendall = calc_kendall
        self.min_obs = min_obs

    def calc_batch(
        self,
        data_list: list[pd.DataFrame],
        gpi_infos: list[tuple],
    ) -> list[dict]:
        """Compute pairwise metrics for a batch of grid points.

        Parameters
        ----------
        data_list : list of pd.DataFrame
            Each DataFrame has 2 columns (x, y) with matched time series.
            May have different lengths (different n_obs per point).
        gpi_infos : list of tuple
            Each tuple is (gpi, lon, lat) or (gpi, lon, lat, metadata_dict).

        Returns
        -------
        list of dict
            One result dict per grid point, in the pytesmo result format.
        """
        backend = get_gpu_backend()
        torch = backend.torch
        n_points = len(data_list)
        results: list[dict | None] = [None] * n_points

        valid_x: list[np.ndarray] = []
        valid_y: list[np.ndarray] = []
        valid_idx: list[int] = []
        for i in range(n_points):
            data = data_list[i].dropna()
            n_obs = len(data)
            if n_obs < self.min_obs:
                results[i] = self._insufficient_result(gpi_infos[i], n_obs)
            else:
                vals = data.values
                valid_x.append(np.asarray(vals[:, 0], dtype=np.float64))
                valid_y.append(np.asarray(vals[:, 1], dtype=np.float64))
                valid_idx.append(i)

        if not valid_idx:
            return results  # type: ignore[return-value]

        n_valid = len(valid_idx)
        max_len = max(len(x) for x in valid_x)
        batch = np.full((n_valid, 2, max_len), np.nan, dtype=np.float64)
        mask = np.zeros((n_valid, max_len), dtype=np.float64)
        for j in range(n_valid):
            t = len(valid_x[j])
            batch[j, 0, :t] = valid_x[j]
            batch[j, 1, :t] = valid_y[j]
            mask[j, :t] = 1.0

        if torch is not None and backend.available:
            x_gpu = backend.to_tensor(batch[:, 0, :])
            y_gpu = backend.to_tensor(batch[:, 1, :])
            mask_gpu = backend.to_tensor(mask)

            n_obs_arr = mask_gpu.sum(dim=1)
            mx = _nanmean_torch(x_gpu, mask_gpu, dim=1)
            my = _nanmean_torch(y_gpu, mask_gpu, dim=1)
            dx = x_gpu - mx.unsqueeze(1)
            dy = y_gpu - my.unsqueeze(1)
            vx = torch.nansum(dx * dx * mask_gpu, dim=1) / n_obs_arr
            vy = torch.nansum(dy * dy * mask_gpu, dim=1) / n_obs_arr
            cov = torch.nansum(dx * dy * mask_gpu, dim=1) / n_obs_arr

            bias = mx - my
            mse_corr = torch.maximum(2.0 * torch.sqrt(vx * vy) - 2.0 * cov, torch.zeros_like(cov))
            mse_var = (torch.sqrt(vx) - torch.sqrt(vy)) ** 2
            mse_bias = bias**2
            mse = mse_corr + mse_var + mse_bias
            rmsd = torch.sqrt(mse)
            urmsd = torch.sqrt(torch.maximum(mse - mse_bias, torch.zeros_like(mse_bias)))
            rss = mse * n_obs_arr
            r = cov / torch.sqrt(vx * vy)

            backend.synchronize()

            n_obs_h = _to_numpy(n_obs_arr).astype(np.int32)
            host_vals = {
                "BIAS": _to_numpy(bias),
                "RMSD": _to_numpy(rmsd),
                "urmsd": _to_numpy(urmsd),
                "mse": _to_numpy(mse),
                "mse_corr": _to_numpy(mse_corr),
                "mse_var": _to_numpy(mse_var),
                "mse_bias": _to_numpy(mse_bias),
                "RSS": _to_numpy(rss),
                "R": _to_numpy(r),
            }

            del x_gpu, y_gpu, mask_gpu, dx, dy, vx, vy, cov
            del bias, mse, rmsd, urmsd, mse_corr, mse_var, mse_bias, rss, r
            del n_obs_arr, mx, my
            backend.empty_cache()
        else:
            x_np = batch[:, 0, :]
            y_np = batch[:, 1, :]
            mask_np = mask

            n_obs_arr = mask_np.sum(axis=1)
            mx = _nanmean_torch(x_np, mask_np, dim=1)
            my = _nanmean_torch(y_np, mask_np, dim=1)
            dx = x_np - mx[:, None]
            dy = y_np - my[:, None]
            vx = np.nansum(dx * dx * mask_np, axis=1) / n_obs_arr
            vy = np.nansum(dy * dy * mask_np, axis=1) / n_obs_arr
            cov = np.nansum(dx * dy * mask_np, axis=1) / n_obs_arr

            bias = mx - my
            mse_corr = np.maximum(2.0 * np.sqrt(vx * vy) - 2.0 * cov, 0.0)
            mse_var = (np.sqrt(vx) - np.sqrt(vy)) ** 2
            mse_bias = bias**2
            mse = mse_corr + mse_var + mse_bias
            rmsd = np.sqrt(mse)
            urmsd = np.sqrt(np.maximum(mse - mse_bias, 0.0))
            rss = mse * n_obs_arr
            r = cov / np.sqrt(vx * vy)

            n_obs_h = n_obs_arr.astype(np.int32)
            host_vals = {
                "BIAS": bias,
                "RMSD": rmsd,
                "urmsd": urmsd,
                "mse": mse,
                "mse_corr": mse_corr,
                "mse_var": mse_var,
                "mse_bias": mse_bias,
                "RSS": rss,
                "R": r,
            }

        for j in range(n_valid):
            i = valid_idx[j]
            x = valid_x[j]
            y = valid_y[j]
            p_r = self._pearson_pvalue(x, y)
            rho, p_rho = self._spearman(x, y) if self.calc_spearman else (np.nan, np.nan)
            tau, p_tau = self._kendall(x, y) if self.calc_kendall else (np.nan, np.nan)
            results[i] = self._build_result(
                gpi_info=gpi_infos[i],
                n_obs=int(n_obs_h[j]),
                vals={k: float(v[j]) for k, v in host_vals.items()},
                p_r=float(p_r),
                rho=float(rho),
                p_rho=float(p_rho),
                tau=float(tau),
                p_tau=float(p_tau),
            )

        return results  # type: ignore[return-value]

    @staticmethod
    def _pearson_pvalue(x: np.ndarray, y: np.ndarray) -> float:
        try:
            return float(stats.pearsonr(x, y)[1])
        except Exception as exc:
            LOGGER.warning("pearsonr p-value failed: %s", exc)
            return float(np.nan)

    def _spearman(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        try:
            rho, p = stats.spearmanr(x, y)
            return float(rho), float(p)
        except Exception as exc:
            LOGGER.warning("spearmanr failed: %s", exc)
            return float(np.nan), float(np.nan)

    def _kendall(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        try:
            tau, p = stats.kendalltau(x, y)
            return float(tau), float(p)
        except Exception as exc:
            LOGGER.warning("kendalltau failed: %s", exc)
            return float(np.nan), float(np.nan)

    def _build_result(
        self,
        gpi_info: tuple,
        n_obs: int,
        vals: dict[str, float],
        p_r: float,
        rho: float,
        p_rho: float,
        tau: float,
        p_tau: float,
    ) -> dict:
        result: dict = {
            "gpi": np.array([gpi_info[0]]),
            "lon": np.array([gpi_info[1]], dtype=np.float64),
            "lat": np.array([gpi_info[2]], dtype=np.float64),
            "n_obs": np.array([n_obs], dtype=np.int32),
            "status": np.array([_STATUS_OK], dtype=np.int32),
        }
        for metric in _ALWAYS_METRICS:
            val = p_r if metric == "p_R" else vals[metric]
            result[metric + _SUFFIX] = np.array([val], dtype=np.float32)
        if self.calc_spearman:
            result["rho" + _SUFFIX] = np.array([rho], dtype=np.float32)
            result["p_rho" + _SUFFIX] = np.array([p_rho], dtype=np.float32)
        if self.calc_kendall:
            result["tau" + _SUFFIX] = np.array([tau], dtype=np.float32)
            result["p_tau" + _SUFFIX] = np.array([p_tau], dtype=np.float32)
        self._add_metadata(result, gpi_info)
        return result

    def _insufficient_result(self, gpi_info: tuple, n_obs: int) -> dict:
        result: dict = {
            "gpi": np.array([gpi_info[0]]),
            "lon": np.array([gpi_info[1]], dtype=np.float64),
            "lat": np.array([gpi_info[2]], dtype=np.float64),
            "n_obs": np.array([n_obs], dtype=np.int32),
            "status": np.array([_STATUS_INSUFFICIENT], dtype=np.int32),
        }
        for metric in _ALWAYS_METRICS:
            result[metric + _SUFFIX] = np.array([np.nan], dtype=np.float32)
        if self.calc_spearman:
            result["rho" + _SUFFIX] = np.array([np.nan], dtype=np.float32)
            result["p_rho" + _SUFFIX] = np.array([np.nan], dtype=np.float32)
        if self.calc_kendall:
            result["tau" + _SUFFIX] = np.array([np.nan], dtype=np.float32)
            result["p_tau" + _SUFFIX] = np.array([np.nan], dtype=np.float32)
        self._add_metadata(result, gpi_info)
        return result

    def _add_metadata(self, result: dict, gpi_info: tuple) -> None:
        if not self.metadata_template:
            return
        meta = gpi_info[3] if len(gpi_info) > 3 else {}
        for key, tmpl in self.metadata_template.items():
            arr = np.array(tmpl, copy=True)
            if key in meta:
                arr[0] = meta[key]
            result[key] = arr


class BatchedTCAMetrics:
    """Compute Triple Collocation metrics for N grid points in a batched GPU pass.

    Computes snr, err_std (scaled) and beta for 3 collocated time series per
    grid point. Result dictionaries use tuple keys like ``("snr", name)`` to
    match the format produced by
    :class:`pytesmo.validation_framework.metric_calculators.TripleCollocationMetrics`.
    """

    def __init__(
        self,
        refname: str,
        dataset_names: list[str],
        min_obs: int = 10,
        metadata_template: dict | None = None,
    ) -> None:
        self.refname = refname
        self.dataset_names = dataset_names
        self.min_obs = min_obs
        self.metadata_template = metadata_template or {}
        self.ref_ind = 0

    def calc_batch(
        self,
        data_list: list[pd.DataFrame],
        gpi_infos: list[tuple],
    ) -> list[dict]:
        """Compute TCA metrics for a batch of grid points.

        Parameters
        ----------
        data_list : list of pd.DataFrame
            Each DataFrame has 3 columns (ref, ds1, ds2) with matched time series.
        gpi_infos : list of tuple
            Each tuple is (gpi, lon, lat) or (gpi, lon, lat, metadata_dict).

        Returns
        -------
        list of dict
            One result dict per grid point, in the pytesmo TCA result format
            with tuple keys like ("snr", "dataset_name").
        """
        backend = get_gpu_backend()
        torch = backend.torch
        n_points = len(data_list)
        results: list[dict | None] = [None] * n_points

        valid_arrays: list[np.ndarray] = []
        valid_idx: list[int] = []
        for i in range(n_points):
            data = data_list[i].dropna()
            n_obs = len(data)
            if n_obs < self.min_obs:
                results[i] = self._insufficient_result(gpi_infos[i], n_obs)
            else:
                vals = np.column_stack([data[name].values for name in self.dataset_names])
                valid_arrays.append(np.asarray(vals, dtype=np.float64))
                valid_idx.append(i)

        if not valid_idx:
            return results  # type: ignore[return-value]

        n_valid = len(valid_idx)
        max_len = max(arr.shape[0] for arr in valid_arrays)
        batch = np.full((n_valid, 3, max_len), np.nan, dtype=np.float64)
        mask = np.zeros((n_valid, max_len), dtype=np.float64)
        for j in range(n_valid):
            t = valid_arrays[j].shape[0]
            batch[j, :, :t] = valid_arrays[j].T
            mask[j, :t] = 1.0

        if torch is not None and backend.available:
            data_gpu = backend.to_tensor(batch)
            mask_gpu = backend.to_tensor(mask)
            snr_host, err_std_host, beta_host, n_obs_h = self._compute_tca_batch_gpu(data_gpu, mask_gpu, n_valid)
        else:
            snr_host, err_std_host, beta_host, n_obs_h = self._compute_tca_batch_numpy(batch, mask)

        for j in range(n_valid):
            i = valid_idx[j]
            results[i] = self._build_result(
                gpi_info=gpi_infos[i],
                n_obs=int(n_obs_h[j]),
                snr=[float(snr_host[k][j]) for k in range(3)],
                err_std=[float(err_std_host[k][j]) for k in range(3)],
                beta=[float(beta_host[k][j]) for k in range(3)],
            )

        return results  # type: ignore[return-value]

    def _compute_tca_batch_gpu(self, data_gpu, mask_gpu, n_valid):
        backend = get_gpu_backend()
        torch = backend.torch

        n_obs_arr = mask_gpu.sum(dim=1)
        means = torch.stack(
            [_nanmean_torch(data_gpu[:, k, :], mask_gpu, dim=1) for k in range(3)],
            dim=1,
        )
        centered = data_gpu - means.unsqueeze(2)

        def _cov(a: int, b: int):
            return torch.nansum(centered[:, a, :] * centered[:, b, :] * mask_gpu, dim=1) / n_obs_arr

        cov_00 = _cov(0, 0)
        cov_11 = _cov(1, 1)
        cov_22 = _cov(2, 2)
        cov_01 = _cov(0, 1)
        cov_02 = _cov(0, 2)
        cov_12 = _cov(1, 2)

        eps = 1e-30

        def _safe_denom(x):
            return torch.where(torch.abs(x) < eps, torch.full_like(x, eps), x)

        ratio_0 = cov_00 * cov_12 / torch.maximum(torch.abs(cov_01 * cov_02), torch.full_like(cov_01, eps))
        ratio_1 = cov_11 * cov_02 / torch.maximum(torch.abs(cov_12 * cov_01), torch.full_like(cov_12, eps))
        ratio_2 = cov_22 * cov_01 / torch.maximum(torch.abs(cov_02 * cov_12), torch.full_like(cov_02, eps))
        snr_0 = 10.0 * torch.log10(
            torch.maximum(1.0 / torch.abs(torch.abs(ratio_0) - 1.0), torch.full_like(ratio_0, eps))
        )
        snr_1 = 10.0 * torch.log10(
            torch.maximum(1.0 / torch.abs(torch.abs(ratio_1) - 1.0), torch.full_like(ratio_1, eps))
        )
        snr_2 = 10.0 * torch.log10(
            torch.maximum(1.0 / torch.abs(torch.abs(ratio_2) - 1.0), torch.full_like(ratio_2, eps))
        )

        err_var_0 = cov_00 - cov_01 * cov_02 / _safe_denom(cov_12)
        err_var_1 = cov_11 - cov_12 * cov_01 / _safe_denom(cov_02)
        err_var_2 = cov_22 - cov_02 * cov_12 / _safe_denom(cov_01)
        err_std_0 = torch.sqrt(torch.maximum(err_var_0, torch.zeros_like(err_var_0)))
        err_std_1 = torch.sqrt(torch.maximum(err_var_1, torch.zeros_like(err_var_1)))
        err_std_2 = torch.sqrt(torch.maximum(err_var_2, torch.zeros_like(err_var_2)))

        beta_1 = cov_02 / _safe_denom(cov_12)
        beta_2 = cov_01 / _safe_denom(cov_12)

        err_std_scaled_0 = err_std_0
        err_std_scaled_1 = err_std_1 * beta_1
        err_std_scaled_2 = err_std_2 * beta_2

        backend.synchronize()

        snr_host = [_to_numpy(snr_0), _to_numpy(snr_1), _to_numpy(snr_2)]
        err_std_host = [
            _to_numpy(err_std_scaled_0),
            _to_numpy(err_std_scaled_1),
            _to_numpy(err_std_scaled_2),
        ]
        beta_host = [
            np.ones(n_valid, dtype=np.float64),
            _to_numpy(beta_1),
            _to_numpy(beta_2),
        ]
        n_obs_h = _to_numpy(n_obs_arr).astype(np.int32)

        del cov_00, cov_11, cov_22, cov_01, cov_02, cov_12
        del snr_0, snr_1, snr_2, ratio_0, ratio_1, ratio_2
        del err_var_0, err_var_1, err_var_2
        del err_std_0, err_std_1, err_std_2
        del err_std_scaled_0, err_std_scaled_1, err_std_scaled_2
        del beta_1, beta_2
        backend.empty_cache()
        return snr_host, err_std_host, beta_host, n_obs_h

    def _compute_tca_batch_numpy(self, batch, mask):
        data_np = batch
        mask_np = mask

        n_obs_arr = mask_np.sum(axis=1)
        means = np.stack(
            [_nanmean_torch(data_np[:, k, :], mask_np, dim=1) for k in range(3)],
            axis=1,
        )
        centered = data_np - means[:, :, None]

        def _cov(a: int, b: int):
            return np.nansum(centered[:, a, :] * centered[:, b, :] * mask_np, axis=1) / n_obs_arr

        cov_00 = _cov(0, 0)
        cov_11 = _cov(1, 1)
        cov_22 = _cov(2, 2)
        cov_01 = _cov(0, 1)
        cov_02 = _cov(0, 2)
        cov_12 = _cov(1, 2)

        eps = 1e-30

        def _safe_denom(x):
            return np.where(np.abs(x) < eps, eps, x)

        ratio_0 = cov_00 * cov_12 / np.maximum(np.abs(cov_01 * cov_02), eps)
        ratio_1 = cov_11 * cov_02 / np.maximum(np.abs(cov_12 * cov_01), eps)
        ratio_2 = cov_22 * cov_01 / np.maximum(np.abs(cov_02 * cov_12), eps)
        snr_0 = 10.0 * np.log10(np.maximum(1.0 / np.abs(np.abs(ratio_0) - 1.0), eps))
        snr_1 = 10.0 * np.log10(np.maximum(1.0 / np.abs(np.abs(ratio_1) - 1.0), eps))
        snr_2 = 10.0 * np.log10(np.maximum(1.0 / np.abs(np.abs(ratio_2) - 1.0), eps))

        err_var_0 = cov_00 - cov_01 * cov_02 / _safe_denom(cov_12)
        err_var_1 = cov_11 - cov_12 * cov_01 / _safe_denom(cov_02)
        err_var_2 = cov_22 - cov_02 * cov_12 / _safe_denom(cov_01)
        err_std_0 = np.sqrt(np.maximum(err_var_0, 0.0))
        err_std_1 = np.sqrt(np.maximum(err_var_1, 0.0))
        err_std_2 = np.sqrt(np.maximum(err_var_2, 0.0))

        beta_1 = cov_02 / _safe_denom(cov_12)
        beta_2 = cov_01 / _safe_denom(cov_12)

        err_std_scaled_0 = err_std_0
        err_std_scaled_1 = err_std_1 * beta_1
        err_std_scaled_2 = err_std_2 * beta_2

        n_valid = batch.shape[0]
        snr_host = [snr_0, snr_1, snr_2]
        err_std_host = [err_std_scaled_0, err_std_scaled_1, err_std_scaled_2]
        beta_host = [
            np.ones(n_valid, dtype=np.float64),
            beta_1,
            beta_2,
        ]
        n_obs_h = n_obs_arr.astype(np.int32)
        return snr_host, err_std_host, beta_host, n_obs_h

    def calc_batch_with_bootstrap(
        self,
        data_list: list[pd.DataFrame],
        gpi_infos: list[tuple],
        nsamples: int = 1000,
        alpha: float = 0.05,
    ) -> list[dict]:
        """Compute TCA metrics with bootstrap confidence intervals.

        Generates ``nsamples`` bootstrap resamples per grid point and computes
        TCA metrics for each resample using batched GPU operations, then derives
        percentile-based confidence intervals for snr, err_std and beta.

        Each result dict contains the point-estimate keys produced by
        :meth:`calc_batch` plus ``("<metric>_ci_lower", name)`` and
        ``("<metric>_ci_upper", name)`` tuple keys for each dataset.
        """
        if not get_gpu_backend().available:
            return self._calc_batch_with_bootstrap_cpu(data_list, gpi_infos, nsamples, alpha)

        results = self.calc_batch(data_list, gpi_infos)
        self._add_bootstrap_cis(results, data_list, nsamples, alpha, gpu=True)
        return results

    def _calc_batch_with_bootstrap_cpu(
        self,
        data_list: list[pd.DataFrame],
        gpi_infos: list[tuple],
        nsamples: int,
        alpha: float,
    ) -> list[dict]:
        """NumPy-based bootstrap CI fallback mirroring the GPU math."""
        results = self.calc_batch(data_list, gpi_infos)
        self._add_bootstrap_cis(results, data_list, nsamples, alpha, gpu=False)
        return results

    def _add_bootstrap_cis(
        self,
        results: Sequence[dict | None],
        data_list: list[pd.DataFrame],
        nsamples: int,
        alpha: float,
        gpu: bool,
    ) -> None:
        """Generate bootstrap CIs and attach them to each valid result dict."""
        backend = get_gpu_backend()
        torch = backend.torch
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        for i in range(len(data_list)):
            res = results[i]
            if res is None:
                continue
            status = res.get("status")
            if status is not None and int(status[0]) == _STATUS_INSUFFICIENT:
                continue

            data = data_list[i].dropna()
            vals = np.column_stack([data[name].values for name in self.dataset_names]).astype(np.float64)
            n = vals.shape[0]
            if n < 1:
                continue

            if gpu and torch is not None and backend.available:
                x_j = backend.to_tensor(vals[:, 0])
                y_j = backend.to_tensor(vals[:, 1])
                z_j = backend.to_tensor(vals[:, 2])
                indices = torch.randint(0, n, (nsamples, n), device=backend.device)
                bx = x_j[indices]
                by = y_j[indices]
                bz = z_j[indices]
                cx = bx - bx.mean(dim=1, keepdim=True)
                cy = by - by.mean(dim=1, keepdim=True)
                cz = bz - bz.mean(dim=1, keepdim=True)
                c00 = (cx * cx).mean(dim=1)
                c11 = (cy * cy).mean(dim=1)
                c22 = (cz * cz).mean(dim=1)
                c01 = (cx * cy).mean(dim=1)
                c02 = (cx * cz).mean(dim=1)
                c12 = (cy * cz).mean(dim=1)
                boot_snr, boot_err_std, boot_beta = self._batched_tca_from_cov_gpu(c00, c11, c22, c01, c02, c12)
                backend.synchronize()
                boot_snr_h = _to_numpy(boot_snr)
                boot_err_std_h = _to_numpy(boot_err_std)
                boot_beta_h = _to_numpy(boot_beta)
                del indices, bx, by, bz, cx, cy, cz
                del boot_snr, boot_err_std, boot_beta
                backend.empty_cache()
            else:
                x_j = vals[:, 0]
                y_j = vals[:, 1]
                z_j = vals[:, 2]
                rng = np.random.default_rng()
                indices = rng.integers(0, n, size=(nsamples, n))
                bx = x_j[indices]
                by = y_j[indices]
                bz = z_j[indices]
                cx = bx - bx.mean(axis=1, keepdims=True)
                cy = by - by.mean(axis=1, keepdims=True)
                cz = bz - bz.mean(axis=1, keepdims=True)
                c00 = (cx * cx).mean(axis=1)
                c11 = (cy * cy).mean(axis=1)
                c22 = (cz * cz).mean(axis=1)
                c01 = (cx * cy).mean(axis=1)
                c02 = (cx * cz).mean(axis=1)
                c12 = (cy * cz).mean(axis=1)
                boot_snr_h, boot_err_std_h, boot_beta_h = self._batched_tca_from_cov_numpy(
                    c00, c11, c22, c01, c02, c12
                )

            snr_lower = np.nanpercentile(boot_snr_h, lower_pct, axis=0)
            snr_upper = np.nanpercentile(boot_snr_h, upper_pct, axis=0)
            err_std_lower = np.nanpercentile(boot_err_std_h, lower_pct, axis=0)
            err_std_upper = np.nanpercentile(boot_err_std_h, upper_pct, axis=0)
            beta_lower = np.nanpercentile(boot_beta_h, lower_pct, axis=0)
            beta_upper = np.nanpercentile(boot_beta_h, upper_pct, axis=0)

            beta_lower[self.ref_ind] = 1.0
            beta_upper[self.ref_ind] = 1.0

            for k, name in enumerate(self.dataset_names):
                res[("snr_ci_lower", name)] = np.array([snr_lower[k]], dtype=np.float32)
                res[("snr_ci_upper", name)] = np.array([snr_upper[k]], dtype=np.float32)
                res[("err_std_ci_lower", name)] = np.array([err_std_lower[k]], dtype=np.float32)
                res[("err_std_ci_upper", name)] = np.array([err_std_upper[k]], dtype=np.float32)
                res[("beta_ci_lower", name)] = np.array([beta_lower[k]], dtype=np.float32)
                res[("beta_ci_upper", name)] = np.array([beta_upper[k]], dtype=np.float32)

    def _batched_tca_from_cov_gpu(self, c00, c11, c22, c01, c02, c12):
        torch = get_gpu_backend().torch
        eps = 1e-30

        def _safe_denom(x):
            return torch.where(torch.abs(x) < eps, torch.full_like(x, eps), x)

        ratio_0 = c00 * c12 / torch.maximum(torch.abs(c01 * c02), torch.full_like(c01, eps))
        ratio_1 = c11 * c02 / torch.maximum(torch.abs(c12 * c01), torch.full_like(c12, eps))
        ratio_2 = c22 * c01 / torch.maximum(torch.abs(c02 * c12), torch.full_like(c02, eps))
        snr_0 = 10.0 * torch.log10(
            torch.maximum(1.0 / torch.abs(torch.abs(ratio_0) - 1.0), torch.full_like(ratio_0, eps))
        )
        snr_1 = 10.0 * torch.log10(
            torch.maximum(1.0 / torch.abs(torch.abs(ratio_1) - 1.0), torch.full_like(ratio_1, eps))
        )
        snr_2 = 10.0 * torch.log10(
            torch.maximum(1.0 / torch.abs(torch.abs(ratio_2) - 1.0), torch.full_like(ratio_2, eps))
        )
        snr = torch.stack([snr_0, snr_1, snr_2], dim=1)

        err_var_0 = c00 - c01 * c02 / _safe_denom(c12)
        err_var_1 = c11 - c12 * c01 / _safe_denom(c02)
        err_var_2 = c22 - c02 * c12 / _safe_denom(c01)
        err_std_0 = torch.sqrt(torch.maximum(err_var_0, torch.zeros_like(err_var_0)))
        err_std_1 = torch.sqrt(torch.maximum(err_var_1, torch.zeros_like(err_var_1)))
        err_std_2 = torch.sqrt(torch.maximum(err_var_2, torch.zeros_like(err_var_2)))

        beta_1 = c02 / _safe_denom(c12)
        beta_2 = c01 / _safe_denom(c12)
        beta = torch.stack([torch.ones_like(c00), beta_1, beta_2], dim=1)

        err_std_scaled = torch.stack([err_std_0, err_std_1 * beta_1, err_std_2 * beta_2], dim=1)

        return snr, err_std_scaled, beta

    def _batched_tca_from_cov_numpy(self, c00, c11, c22, c01, c02, c12):
        eps = 1e-30

        def _safe_denom(x):
            return np.where(np.abs(x) < eps, eps, x)

        ratio_0 = c00 * c12 / np.maximum(np.abs(c01 * c02), eps)
        ratio_1 = c11 * c02 / np.maximum(np.abs(c12 * c01), eps)
        ratio_2 = c22 * c01 / np.maximum(np.abs(c02 * c12), eps)
        snr_0 = 10.0 * np.log10(np.maximum(1.0 / np.abs(np.abs(ratio_0) - 1.0), eps))
        snr_1 = 10.0 * np.log10(np.maximum(1.0 / np.abs(np.abs(ratio_1) - 1.0), eps))
        snr_2 = 10.0 * np.log10(np.maximum(1.0 / np.abs(np.abs(ratio_2) - 1.0), eps))
        snr = np.stack([snr_0, snr_1, snr_2], axis=1)

        err_var_0 = c00 - c01 * c02 / _safe_denom(c12)
        err_var_1 = c11 - c12 * c01 / _safe_denom(c02)
        err_var_2 = c22 - c02 * c12 / _safe_denom(c01)
        err_std_0 = np.sqrt(np.maximum(err_var_0, 0.0))
        err_std_1 = np.sqrt(np.maximum(err_var_1, 0.0))
        err_std_2 = np.sqrt(np.maximum(err_var_2, 0.0))

        beta_1 = c02 / _safe_denom(c12)
        beta_2 = c01 / _safe_denom(c12)
        beta = np.stack([np.ones_like(c00), beta_1, beta_2], axis=1)

        err_std_scaled = np.stack([err_std_0, err_std_1 * beta_1, err_std_2 * beta_2], axis=1)

        return snr, err_std_scaled, beta

    def _build_result(
        self,
        gpi_info: tuple,
        n_obs: int,
        snr: list[float],
        err_std: list[float],
        beta: list[float],
    ) -> dict:
        result: dict = {
            "gpi": np.array([gpi_info[0]]),
            "lon": np.array([gpi_info[1]], dtype=np.float64),
            "lat": np.array([gpi_info[2]], dtype=np.float64),
            "n_obs": np.array([n_obs], dtype=np.int32),
            "status": np.array([_STATUS_OK], dtype=np.int32),
        }
        for i, name in enumerate(self.dataset_names):
            result[("snr", name)] = np.array([snr[i]], dtype=np.float32)
            result[("err_std", name)] = np.array([err_std[i]], dtype=np.float32)
            result[("beta", name)] = np.array([beta[i]], dtype=np.float32)
        self._add_metadata(result, gpi_info)
        return result

    def _insufficient_result(self, gpi_info: tuple, n_obs: int) -> dict:
        result: dict = {
            "gpi": np.array([gpi_info[0]]),
            "lon": np.array([gpi_info[1]], dtype=np.float64),
            "lat": np.array([gpi_info[2]], dtype=np.float64),
            "n_obs": np.array([n_obs], dtype=np.int32),
            "status": np.array([_STATUS_INSUFFICIENT], dtype=np.int32),
        }
        for name in self.dataset_names:
            result[("snr", name)] = np.array([np.nan], dtype=np.float32)
            result[("err_std", name)] = np.array([np.nan], dtype=np.float32)
            result[("beta", name)] = np.array([np.nan], dtype=np.float32)
        self._add_metadata(result, gpi_info)
        return result

    def _add_metadata(self, result: dict, gpi_info: tuple) -> None:
        if not self.metadata_template:
            return
        meta = gpi_info[3] if len(gpi_info) > 3 else {}
        for key, tmpl in self.metadata_template.items():
            arr = np.array(tmpl, copy=True)
            if key in meta:
                arr[0] = meta[key]
            result[key] = arr


class GPUBatchedValidation:
    """GPU-accelerated batched validation that replaces pytesmo's ``Validation.calc()``.

    Collects matched time series for all grid points in a job, then computes
    metrics in a single batched GPU pass. The ``calc()`` method returns results
    in the same format as ``pytesmo.validation_framework.validation.Validation.calc()``.
    """

    def __init__(
        self,
        datamanager,
        temporal_matcher,
        temporal_ref: str,
        spatial_ref: str,
        scaling: str | None,
        scaling_ref: str | None,
        metrics_calculators: dict,
        ds_names: list[str],
        metadata_template: dict | None = None,
        calc_kendall: bool = False,
        tcol_enabled: bool = False,
        bootstrap_tcol_cis: bool = False,
        dataset_short_names: list[str] | None = None,
    ) -> None:
        self.datamanager = datamanager
        self.temp_matching = temporal_matcher
        self.temporal_ref = temporal_ref or datamanager.reference_name
        self.spatial_ref = spatial_ref
        self.metrics_c = metrics_calculators
        self.ds_names = ds_names
        self.metadata_template = metadata_template or {}
        self.dataset_short_names = dataset_short_names or []
        self.tcol_enabled = tcol_enabled
        self.bootstrap_tcol_cis = bootstrap_tcol_cis

        if isinstance(scaling, str):
            self.scaling: Any = DefaultScaler(scaling)
        else:
            self.scaling = scaling
        self.scaling_ref = scaling_ref or datamanager.reference_name

        self.pairwise_calc = BatchedPairwiseMetrics(
            metadata_template=metadata_template,
            calc_kendall=calc_kendall,
            min_obs=10,
        )

        self.tcol_calc: BatchedTCAMetrics | None = None
        if tcol_enabled and len(datamanager.datasets) >= 3:
            ref_ds = datamanager.reference_name
            other_ds = [n for n in datamanager.datasets if n != ref_ds][:2]
            tca_dataset_names = [ref_ds] + other_ds
            self.tcol_calc = BatchedTCAMetrics(
                refname=ref_ds,
                dataset_names=tca_dataset_names,
                min_obs=10,
                metadata_template=metadata_template,
            )

    def calc(
        self,
        gpis,
        lons,
        lats,
        meta_list=None,
        only_with_reference: bool = True,
        handle_errors: str = "ignore",
        **kwargs,
    ) -> dict:
        """Compute metrics for a batch of grid points.

        Returns results in the same format as ``Validation.calc()``: a dict
        keyed by tuples of ``(dataset_name, column_name)`` pairs, with values
        being dicts of ``metric_name -> np.ndarray`` (one entry per grid point).
        """
        gpis = list(gpis)
        lons = list(lons)
        lats = list(lats)

        result_keys_by_k: dict[int, list] = {}
        for n, k in self.metrics_c:
            if k in result_keys_by_k:
                continue
            names = get_result_combinations(self.datamanager.ds_dict, n=k)
            if only_with_reference:
                names = [rk for rk in names if self.datamanager.reference_name in [r[0] for r in rk]]
            result_keys_by_k[k] = names

        has_pairwise = 2 in result_keys_by_k
        has_tcol = 3 in result_keys_by_k and self.tcol_calc is not None

        pairwise_data: dict[tuple, list[pd.DataFrame]] = {}
        tcol_data: dict[tuple, list[pd.DataFrame]] = {}
        for k, keys in result_keys_by_k.items():
            for rk in keys:
                if k == 2 and has_pairwise:
                    pairwise_data[rk] = []
                elif k == 3 and has_tcol:
                    tcol_data[rk] = []

        gpi_infos: list[tuple] = []

        for i, gpi in enumerate(gpis):
            gpi_info = self._make_gpi_info(gpi, lons[i], lats[i], meta_list, i)
            gpi_infos.append(gpi_info)

            df_dict = self._read_data(gpi, lons[i], lats[i])
            if not df_dict:
                self._append_empty(pairwise_data, tcol_data)
                continue

            data_df_dict = self._extract_data_columns(df_dict)

            matched_by_k: dict[int, dict] = {}
            for n, k in self.metrics_c:
                if k not in result_keys_by_k or not result_keys_by_k[k]:
                    continue
                try:
                    matched = self.temp_matching(data_df_dict, self.temporal_ref, n=n, k=k)
                except Exception as exc:
                    LOGGER.warning("Temporal matching failed for gpi %s: %s", gpi, exc)
                    matched = {}
                matched_by_k[k] = matched

            for k, result_keys in result_keys_by_k.items():
                matched = matched_by_k.get(k, {})
                for rk in result_keys:
                    data = self._extract_and_scale(matched, rk, gpi_info)
                    if k == 2 and has_pairwise:
                        pairwise_data[rk].append(data)
                    elif k == 3 and has_tcol:
                        tcol_data[rk].append(data)

        results: dict = {}
        for rk, data_list in pairwise_data.items():
            batched = self.pairwise_calc.calc_batch(data_list, gpi_infos)
            agg = self._aggregate_batched(batched)
            if agg:
                results[rk] = agg
            del batched
        del pairwise_data

        if has_tcol and self.tcol_calc is not None:
            for rk, data_list in tcol_data.items():
                if self.bootstrap_tcol_cis:
                    batched = self.tcol_calc.calc_batch_with_bootstrap(data_list, gpi_infos)
                else:
                    batched = self.tcol_calc.calc_batch(data_list, gpi_infos)
                agg = self._aggregate_batched(batched)
                if agg:
                    if rk in results:
                        results[rk].update(agg)
                    else:
                        results[rk] = agg
                del batched
        del tcol_data

        return results

    @staticmethod
    def _make_gpi_info(gpi, lon, lat, meta_list, idx) -> tuple:
        if meta_list is not None and idx < len(meta_list):
            return (gpi, lon, lat, meta_list[idx])
        return (gpi, lon, lat)

    def _read_data(self, gpi, lon, lat) -> dict:
        # Serialize HDF5/netCDF4 reads across worker threads to avoid the
        # Windows 0xC0000005 access violation.
        from qa4sm_gpu_validation.read_lock import HDF5_READ_LOCK

        with HDF5_READ_LOCK:
            try:
                return self.datamanager.get_data(gpi, lon, lat)
            except Exception as exc:
                LOGGER.warning("Failed to read data for gpi %s: %s", gpi, exc)
                return {}

    def _extract_data_columns(self, df_dict: dict) -> dict:
        data_df_dict = {}
        for ds, df in df_dict.items():
            columns = self.datamanager.datasets[ds]["columns"]
            data_df_dict[ds] = df[columns]
        return data_df_dict

    def _extract_and_scale(self, matched: dict, result_key: tuple, gpi_info: tuple) -> pd.DataFrame:
        """Extract k columns for *result_key* from matched data, then scale."""
        if not matched:
            return pd.DataFrame()

        matched_df = next(iter(matched.values()))

        cols = list(result_key)
        if self.scaling is not None:
            scaling_col = (
                self.scaling_ref,
                self.datamanager.datasets[self.scaling_ref]["columns"][0],
            )
            if scaling_col not in cols:
                cols.append(scaling_col)

        available = [c for c in cols if c in matched_df.columns]
        if not available:
            return pd.DataFrame()

        data = matched_df[available]
        data = data.dropna()
        if len(data) == 0:
            return pd.DataFrame()

        data = data.rename(columns=lambda x: x[0])

        if self.scaling is not None and self.scaling_ref in data.columns:
            scaling_index = data.columns.tolist().index(self.scaling_ref)
            try:
                data = self.scaling.scale(data, scaling_index, gpi_info)
            except Exception as exc:
                LOGGER.warning("Scaling failed for gpi %s: %s", gpi_info[0], exc)
                return pd.DataFrame()
            if self.scaling_ref not in [r[0] for r in result_key]:
                data = data.drop(columns=[self.scaling_ref])

        return data

    @staticmethod
    def _append_empty(
        pairwise_data: dict[tuple, list],
        tcol_data: dict[tuple, list],
    ) -> None:
        for rk in pairwise_data:
            pairwise_data[rk].append(pd.DataFrame())
        for rk in tcol_data:
            tcol_data[rk].append(pd.DataFrame())

    @staticmethod
    def _aggregate_batched(batched_results: list) -> dict:
        """Stack per-gpi result dicts into the pytesmo compact format."""
        if not batched_results:
            return {}

        first = next((r for r in batched_results if r is not None), None)
        if first is None:
            return {}

        aggregated: dict = {}
        for field_name in first:
            values = []
            for r in batched_results:
                if r is not None and field_name in r:
                    values.append(r[field_name][0])
                elif field_name in ("n_obs", "status"):
                    values.append(0)
                else:
                    values.append(np.nan)
            aggregated[field_name] = np.array(values)
        return aggregated
