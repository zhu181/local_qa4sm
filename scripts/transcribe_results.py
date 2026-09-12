"""
Standalone script to transcribe and compress pytesmo validation results.

Usage:
    uv run python scripts/transcribe_results.py <input.nc> \
        [--output-dir OUTPUT_DIR] [--keep-pytesmo] \
        [--stability START_YEAR END_YEAR]

This script wraps Pytesmo2Qa4smResultsTranscriber to:
1. Transcribe the pytesmo results to QA4SM format
2. Write the transcribed dataset to a new NetCDF file
3. Compress the output file with zlib

The key fix vs. the original _post_process_run flow: this script explicitly
calls get_transcribed_dataset() before write_to_netcdf(), which initializes
the self.transcribed_dataset attribute that write_to_netcdf() relies on.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from qa4sm_reader.intra_annual_temp_windows import TemporalSubWindowsFactory
from qa4sm_reader.netcdf_transcription import Pytesmo2Qa4smResultsTranscriber


def parse_dataset_keys_from_filename(filename: str):
    """
    Parse dataset configuration keys from the pytesmo results filename.

    Filenames follow the pattern:
        0-ISMN.soil_moisture_with_1-SPL3SMPE.soil_moisture_with_2-NSMCSMC.soil_moisture.nc

    Returns a list containing a single tuple of dataset name strings, matching
    the format used by pytesmo results_manager keys, e.g.:
    [('0-ISMN.soil_moisture', '1-SPL3SMPE.soil_moisture', '2-NSMCSMC.soil_moisture')]
    """
    stem = Path(filename).stem
    parts = re.split(r"_with_", stem)
    # Each part is like "0-ISMN.soil_moisture" — keep as-is
    return [tuple(parts)]


def main():
    parser = argparse.ArgumentParser(description="Transcribe pytesmo validation results to QA4SM NetCDF format.")
    parser.add_argument(
        "input",
        type=str,
        help="Path to the pytesmo results NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the transcribed NetCDF file. Defaults to the same directory as the input file.",
    )
    parser.add_argument(
        "--keep-pytesmo",
        action="store_true",
        help="Keep the original pytesmo NetCDF file after transcription.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Compress the output file with zlib (default: True).",
    )
    parser.add_argument(
        "--complevel",
        type=int,
        default=9,
        help="Compression level (1-9). Default: 9.",
    )
    parser.add_argument(
        "--stability",
        type=int,
        nargs=2,
        metavar=("START_YEAR", "END_YEAR"),
        default=None,
        help="Create stability temporal sub-windows for the given year range. e.g. --stability 2016 2021",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")

    # Build temporal sub-windows instance if requested
    intra_annual_slices = None
    if args.stability:
        start_year, end_year = args.stability
        period = [datetime(start_year, 1, 1), datetime(end_year, 12, 31)]
        intra_annual_slices = TemporalSubWindowsFactory.create(
            temporal_sub_window_type="stability",
            overlap=0,
            period=period,
        )
        print(f"Created stability temporal sub-windows: {intra_annual_slices.names}")

    # Create the transcriber
    print("Creating transcriber...")
    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=str(input_path),
        intra_annual_slices=intra_annual_slices,
        keep_pytesmo_ncfile=args.keep_pytesmo,
    )

    if not transcriber.exists:
        print("Error: transcriber reports the file does not exist.", file=sys.stderr)
        return 1

    # Parse dataset keys from filename for building the output name
    keys = parse_dataset_keys_from_filename(input_path.name)
    print(f"Parsed dataset keys: {keys}")

    # Build output names
    outname, outname_zarr = transcriber.build_outname(str(output_dir), keys)
    transcriber.output_file_name = str(outname)
    transcriber.output_zarr_name = str(outname_zarr)
    print(f"Output NetCDF: {outname}")
    print(f"Output Zarr: {outname_zarr}")

    # KEY FIX: call get_transcribed_dataset() to initialize self.transcribed_dataset
    # before calling write_to_netcdf()
    print("Transcribing dataset...")
    transcriber.get_transcribed_dataset()
    print("Transcription complete.")

    # Write to NetCDF
    print(f"Writing to {outname}...")
    transcriber.write_to_netcdf(transcriber.output_file_name)
    print("NetCDF write complete.")

    # Compress
    if args.compress:
        print(f"Compressing with zlib level {args.complevel}...")
        transcriber.compress(
            path=transcriber.output_file_name,
            compression="zlib",
            complevel=args.complevel,
        )
        print("Compression complete.")

    print(f"\nDone! Output file: {transcriber.output_file_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
