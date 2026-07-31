"""Global lock serializing concurrent HDF5/netCDF4 file reads.

The netCDF4/HDF5 C libraries are not safe for concurrent opens/reads of the
same file from multiple threads (on Windows this can crash with
0xC0000005). :meth:`qa4sm_gpu_validation.gpu_metrics.GPUBatchedValidation._read_data`
acquires this lock only for the ``DataManager.get_data`` read call, so
subsequent GPU/NumPy compute stays fully parallel across worker threads.
"""

import threading

HDF5_READ_LOCK = threading.Lock()
