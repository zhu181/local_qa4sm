from validator.orchestrator import parse_validation_run_config
from validator.graphics import plot_all
import json


if __name__ == "__main__":
    nc_file = r"outputs\0-ISMN.soil_moisture_with_1-SPL3SMP3.soil_moisture_with_2-SMMerge.soil_moisture.nc"
    output_dir = r"outputs\plots"
    plot_all(nc_file, save_metadata="always", out_dir=output_dir)
