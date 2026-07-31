# qa4sm-gpu-validation

Thread-safe, GPU-accelerated, parallel-supported metric calculators that are
drop-in compatible with [pytesmo](https://github.com/TUW-GEO/pytesmo)'s
validation framework.

## Features

- **GPU acceleration** — pairwise (BIAS, RMSD, R, p_R, MSE family, spearman/kendall)
  and triple collocation (snr, err_std, beta, optional bootstrap CIs) metrics for
  N grid points in a single batched PyTorch/CUDA pass.
- **NumPy fallback** — automatic CPU fallback when CUDA or PyTorch is unavailable.
- **pytesmo compatible** — `GPUBatchedValidation.calc()` matches the signature
  and result format of `pytesmo.validation_framework.validation.Validation.calc()`,
  and reuses pytesmo's `get_result_combinations` and `DefaultScaler`.
- **Thread safe** — HDF5/netCDF4 reads are serialized via the global
  `HDF5_READ_LOCK` while GPU/NumPy compute runs in parallel across worker threads.

## Install

```bash
pip install -e ".[gpu]"        # with GPU support (torch)
pip install -e .               # CPU-only (NumPy fallback)
```

Torch on CUDA: configure the appropriate wheel index for your CUDA version
(e.g. `--index-url https://download.pytorch.org/whl/cu132`).

## Usage

```python
from qa4sm_gpu_validation import GPUBatchedValidation, get_gpu_backend

get_gpu_backend()  # selects device from QA4SM_GPU_DEVICE_ID (default 0)

val = GPUBatchedValidation(
    datamanager=data_manager,        # pytesmo DataManager
    temporal_matcher=temporal_matcher,
    temporal_ref=temporal_ref_name,
    spatial_ref=spatial_ref_name,
    scaling=scaling_method,
    scaling_ref=scaling_ref_name,
    metrics_calculators=metrics_calculators,
    ds_names=ds_names,
)
results = val.calc(gpis, lons, lats, only_with_reference=True, handle_errors="ignore")
```

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `QA4SM_GPU_DEVICE_ID` | `0` | CUDA device index for the backend singleton |

## Tests

```bash
uv run pytest
```
