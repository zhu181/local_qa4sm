import logging
from os import listdir, path
from netCDF4 import Dataset as NetCDFDataset

from ismn.interface import ISMN_Interface
from ismn.custom import CustomSensorMetadataCsv
from smap_io.interface import SMAPTs, SMAPL3_V9Reader, ReaderWithExtension_SMAP
from pynetcf.time_series import GriddedNcTs, GriddedNcIndexedRaggedTs
from qa4sm_preprocessing.reading import (
    GriddedNcOrthoMultiTs,
    GriddedNcContiguousRaggedTs,
)

from validator import globals

from pytesmo.validation_framework.adapters import TimestampAdapter
import pandas as pd

from validator.models import Dataset

__logger = logging.getLogger(__name__)


def _uses_indexed_ragged_format(ts_path: str) -> bool:
    try:
        netcdf_files = sorted(
            filename
            for filename in listdir(ts_path)
            if filename.endswith('.nc') and filename != 'grid.nc'
        )
        if not netcdf_files:
            return False

        sample_file = path.join(ts_path, netcdf_files[0])
        with NetCDFDataset(sample_file, 'r') as ds:
            if 'locationIndex' not in ds.variables:
                return False
            soil_moisture = ds.variables.get('soil_moisture')
            if soil_moisture is None:
                return True
            return len(soil_moisture.dimensions) == 1
    except Exception as exc:
        __logger.warning(
            'Failed to inspect SMAP file structure in %s: %s', ts_path, exc
        )
        return False


class ReaderWithTsExtension:
    """
    Concatenate 2 time series upon reading
    """

    def __init__(self, cls, path, path_ext, *args, **kwargs):
        """
        Parameters
        ----------
        cls: Callable
            Reader class to wrap
        path: str
            Path to the main time series (not the extension dataset)
        path_ext: str
            Extension time series path
        args, kwargs:
            Additional arguments to set up the readers
        """
        self.base_reader = cls(path, *args, **kwargs)
        try:
            self.ext_reader = cls(path_ext, *args, **kwargs)
        except FileNotFoundError:
            logging.error(f"No extension dataset found in path {path_ext}")
            self.ext_reader = None

    @property
    def grid(self):
        return self.base_reader.grid

    def read(self, *args, **kwargs) -> pd.DataFrame:
        """
        Read time series at location for both the base dataset and the
        extension. If extension is read, concatenate both in time.
        """
        ts = self.base_reader.read(*args, **kwargs)
        try:
            if self.ext_reader is not None:
                ext = self.ext_reader.read(*args, **kwargs)
                ts = pd.concat([ts, ext], axis=0)
                ts = ts[~ts.index.duplicated(keep="last")]  # prefer ext. data
        except Exception as e:
            logging.error(
                f"Extension reading failed for {args} {kwargs} witherror: {e}"
            )

        return ts


class SBPCAReader(GriddedNcOrthoMultiTs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self, *args, **kwargs) -> pd.DataFrame:
        ts = super().read(*args, **kwargs)
        if (ts is not None) and not ts.empty:
            ts = ts[ts.index.notnull()]
            for col in [
                "Chi_2_P",
                "M_AVA0",
                "N_RFI_X",
                "N_RFI_Y",
                "RFI_Prob",
                "Science_Flags",
            ]:
                if col in ts.columns:
                    ts[col] = ts[col].fillna(0)
            if "Soil_Moisture" in ts.columns:
                ts = ts.dropna(subset="Soil_Moisture")
            if "acquisition_time" in ts.columns:
                ts = ts.dropna(subset="acquisition_time")
        return ts


class SMOSL2Reader(GriddedNcIndexedRaggedTs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self, *args, **kwargs) -> pd.DataFrame:
        ts = super().read(*args, **kwargs)
        if (ts is not None) and not ts.empty:
            ts = ts[ts.index.notnull()]
            for col in [
                "Chi_2_P",
                "M_AVA0",
                "N_RFI_X",
                "N_RFI_Y",
                "RFI_Prob",
                "Science_Flags",
            ]:
                if col in ts.columns:
                    ts[col] = ts[col].fillna(0)
            if "Soil_Moisture" in ts.columns:
                ts = ts.dropna(subset="Soil_Moisture")
            if "acquisition_time" in ts.columns:
                ts = ts.dropna(subset="acquisition_time")
        return ts


def create_reader(dataset:Dataset, version) -> GriddedNcTs:
    """
    Create basic readers (without any adapters / filters) for a dataset version
    """

    reader = None

    if dataset.reader == "ISMN_Interface":
        if path.isfile(path.join(dataset.storage_path, "frm_classification.csv")):
            custom_meta_readers = [
                CustomSensorMetadataCsv(
                    path.join(dataset.storage_path, "frm_classification.csv"),
                    fill_values={"frm_class": "undeducible"},
                ),
            ]
        else:
            custom_meta_readers = None
        reader = ISMN_Interface(dataset.storage_path, custom_meta_reader=custom_meta_readers)

    if dataset.reader == "SMAPTs":
        if _uses_indexed_ragged_format(dataset.storage_path):
            __logger.warning(
                "Dataset %s configured as SMAPTs but files are indexed ragged; falling back to SMAPL3_V9Reader.",
                dataset.short_name,
            )
            reader = SMAPL3_V9Reader(
                dataset.storage_path, ioclass_kws={"read_bulk": True}
            )
        else:
            reader = SMAPTs(dataset.storage_path, ioclass_kws={"read_bulk": True})

    if dataset.reader == "SMAPL3_V9Reader":
        reader = SMAPL3_V9Reader(dataset.storage_path, ioclass_kws={"read_bulk": True})

    if dataset.reader == "GriddedNcOrthoMultiTs":
        reader = GriddedNcOrthoMultiTs(dataset.storage_path, ioclass_kws={"read_bulk": True})
    
    if dataset.reader == "GriddedNcContiguousRaggedTs":
        reader = GriddedNcContiguousRaggedTs(dataset.storage_path, ioclass_kws={"read_bulk": True})

    if not reader:
        raise ValueError("Reader for dataset '{}' not available".format(dataset))

    return reader


def adapt_timestamp(reader, dataset, version):
    """Adapt the reader to include the specified time offset"""
    if dataset.short_name == globals.SMOS_L3:
        tadapt_kwargs = {
            "time_offset_fields": "Mean_Acq_Time_Seconds",
            "time_units": "s",
            "base_time_field": "Mean_Acq_Time_Days",
            "base_time_reference": "2000-01-01",
        }

    elif (dataset.short_name == globals.SMOS_SBPCA) and (
        version.short_name == globals.SMOS_SBPCA_v724
    ):
        tadapt_kwargs = {
            "base_time_field": "acquisition_time",
            "base_time_reference": "2000-01-01T00:00:00",
            "base_time_units": "s",
            "time_offset_fields": None,
            "time_units": None,
        }

    elif dataset.short_name == globals.SMOS_IC:
        tadapt_kwargs = {
            "time_offset_fields": ["UTC_Seconds", "UTC_Microseconds"],
            "time_units": ["s", "us"],
            "base_time_field": "Days",
            "base_time_reference": "2000-01-01",
        }

    elif dataset.short_name == globals.SMAP_L2:
        tadapt_kwargs = {
            "base_time_field": "acquisition_time",
            "base_time_reference": "2000-01-01T12:00:00",
            "base_time_units": "s",
            "time_offset_fields": None,
            "time_units": None,
        }

    # No adaptation needed
    else:
        return reader

    __logger.debug(f"{dataset.short_name} adapted to account for the time offset")

    # Adapt the reader with the exact timestamps
    return TimestampAdapter(reader, **tadapt_kwargs)
