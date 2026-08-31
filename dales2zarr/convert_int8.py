#!/usr/bin/env python

import argparse
import logging
import os
import xarray as xr
import zarr
import yaml
from dales2zarr.zarr_cast import cast_to_int8

DEFAULT_INPUT_CONFIG = {
    "ql":       {"mode": "log",    "file": "fielddump-ql.nc", "variable": "ql"},
    "qr":       {"mode": "linear", "file": "fielddump-qr.nc", "variable": "sv002"},
    "thetavmix":{"mode": "linear", "file": "cape-thetavmix.nc", "variable": "thetavmix"},
}

# Parse command-line arguments
def parse_args(arg_list=None):
    """Parse command-line arguments for the convert_int8 script.

    Args:
        arg_list (list, optional): List of command-line arguments of type str. Defaults to None,
                                    in which case sys.argv[1:] is used.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert input dataset to 8-bit integers and write to zarr")
    parser.add_argument("--input-dir", metavar="DIR", type=str, required=True, dest="input_dir",
                        help="Path to the directory containing input NetCDF files")
    parser.add_argument("--output", metavar="FILE", type=str, required=False, default=None,
                        help="Path to the output zarr file")
    parser.add_argument("--config", metavar="FILE", type=str, required=False, default=None,
                        help="Path to the input configuration file (yaml)")
    parser.add_argument("--levels", metavar="INT", type=int, required=False, default=0, 
                        help="Number of coarsening levels")
    parser.add_argument("--timestamps" , metavar="INT", type=int, required=False, default=0,
                        help="Number of timestamps to keep")
    parser.add_argument("--mode", metavar="w|a", type=str, required=False, default="a", choices=["w", "a"],
                        help="Write or append mode")
    parser.add_argument("--coarsen", metavar="mean|max|median", type=str, required=False, default="mean",
                        choices=["mean", "max", "median"],
                        help="Coarsening method for LOD levels")
    return parser.parse_args(args=arg_list)


def main(arg_list=None):
    """Convert the input dataset to int8 and save it in zarr format.

    Args:
        arg_list (list, optional): List of command-line arguments. Defaults to None, in which case sys.argv[1:] is used.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)
    # Parse command-line arguments
    args = parse_args(arg_list)

    if args.config is None:
        input_config = DEFAULT_INPUT_CONFIG
    else:
        # Read the input configuration from yaml
        with open(args.config, "r") as f:
            input_config = yaml.safe_load(f)

    # Open per-variable datasets from their respective input files
    datasets = {}
    for target_variable, var_options in input_config.items():
        prefix = "cape-" if target_variable == "thetavmix" else "fielddump-"
        filename = var_options.get("file", f"{prefix}{target_variable}.nc")
        filepath = os.path.join(args.input_dir, filename)
        try:
            ds = xr.open_dataset(filepath, chunks='auto')
            if args.timestamps > 0:
                ds = ds.isel(time=slice(0, args.timestamps))
            datasets[target_variable] = ds
        except FileNotFoundError:
            logging.warning(f'Input file {filepath} not found for variable {target_variable}... skipping')

    outfile = args.output if args.output is not None else os.path.join(args.input_dir, "output_int8.zarr")

    # Write the result to zarr with Blosc compression
    compressor = zarr.Blosc(cname="lz4", clevel=6, shuffle=zarr.Blosc.BITSHUFFLE)
    var_encoding = {"dtype": "uint8", "compressor": compressor}

    # Process and write each variable immediately to avoid accumulating all data in memory
    output_variables = []
    write_mode = args.mode
    for target_variable, var_options in input_config.items():
        ds = datasets.get(target_variable)
        if ds is None:
            continue
        source_variable = var_options.get('variable', target_variable)
        cast_option = var_options.get('mode', 'linear')
        int8_var = cast_to_int8(ds, source_variable, target_variable, cast_option)
        if int8_var is not None:
            int8_var = int8_var.chunk({dim: "auto" for dim in int8_var.dims})
            int8_var.to_zarr(outfile, mode=write_mode, encoding={target_variable: var_encoding})
            output_variables.append(target_variable)
            write_mode = "a"

    # Coarsen the dataset and write to zarr
    # NOTE: coarsening assumes all output variables share the same horizontal grid resolution
    ds = xr.open_zarr(outfile) if args.levels > 0 else None
    for level in range(1, args.levels + 1):
        coarsen_op = ds.coarsen({dim: 2 for dim in ds.dims if dim != "time"}, boundary="trim")
        ds = getattr(coarsen_op, args.coarsen)()
        level_outfile = outfile.replace(".zarr", f"-{level}.zarr")
        for var in output_variables:
            var_da = ds[var]
            # use first-chunk sizes to guarantee the last chunk is never larger
            uniform_chunks = {dim: var_da.chunks[i][0] for i, dim in enumerate(var_da.dims)}
            var_da.chunk(uniform_chunks).to_zarr(level_outfile, mode="a", encoding={var: var_encoding})


if __name__ == "__main__":
    main()
