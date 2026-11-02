#!/usr/bin/env python

import argparse
import logging
import math
import os
import xarray as xr
import zarr
import yaml
import numpy as np
from dales2zarr.zarr_cast import cast_to_int8, scale_bounds

DALES_INPUT_CONFIG = {
    "variables": {
        "ql":       {"mode": "log",    "file": "fielddump-ql.nc", "variable": "ql"},
        "qr":       {"mode": "linear", "file": "fielddump-qr.nc", "variable": "sv002"},
        "thetavmix":{"mode": "linear", "file": "cape-thetavmix.nc", "variable": "thetavmix"},
    },
    "dimensions": {"xt": "x", "yt": "y", "zt": "z"}
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
    parser.add_argument("--input-dir", metavar="DIR", type=str, required=False, dest="input_dir",
                        help="Path to the directory containing input NetCDF files")
    parser.add_argument("--input-url", metavar="URL", type=str, required=False, dest="input_url",
                        help="URL to the input dataset")
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
    parser.add_argument("--horizontal-chunks", metavar="INT", type=int, required=False, default=1,
                        dest="horizontal_chunks",
                        help="Number of chunks along each horizontal axis (time chunk is 1, z is unchunked)")
    return parser.parse_args(args=arg_list)


def output_chunks(da, xy_chunks):
    """Chunk sizes: 1 along time, `xy_chunks[dim]` along x/y, full size along all other (z) dimensions."""
    chunks = {}
    for dim in da.dims:
        if dim == "time":
            chunks[dim] = 1
        elif dim in xy_chunks:
            chunks[dim] = xy_chunks[dim]
        else:
            chunks[dim] = da.sizes[dim]
    return chunks


# Zarr encoding: Blosc-compressed uint8
VAR_ENCODING = {"dtype": "uint8", "compressor": zarr.Blosc(cname="lz4", clevel=6, shuffle=zarr.Blosc.BITSHUFFLE)}


def load_datasets(args, input_config):
    """Open per-variable datasets from their input files; returns a dict keyed by target variable."""
    # Open per-variable datasets from their respective input files
    datasets = {}
    for target_variable, var_options in input_config.get("variables", {}).items():
        filename = var_options.get("file", var_options.get("files", None))
        if filename is None:
            logging.warning(f'No input file specified for variable {target_variable}... skipping')
            continue
        if args.input_url:
            filepath = args.input_url + '/' + filename
        else:
            filepath = os.path.join(args.input_dir, filename)
        try:
            if var_options.get("files", None) is not None:
                ds = xr.open_mfdataset(filepath, chunks='auto', engine=var_options.get("engine", None), concat_dim=var_options.get("concat_dim", None), combine='nested' )
            else:
                ds = xr.open_dataset(filepath, chunks='auto', engine=var_options.get("engine", None), missing_values=var_options.get("missing_value", None))
            if args.timestamps > 0:
                ds = ds.isel(time=slice(0, args.timestamps))
            datasets[target_variable] = ds.rename_dims(input_config.get("dimensions", {}))
        except FileNotFoundError:
            logging.warning(f'Input file {filepath} not found for variable {target_variable}... skipping')
    return datasets


def write_int8_dataset(datasets, input_config, outfile, mode, horizontal_chunks):
    """Cast each variable to int8 and write it to `outfile` (zarr).

    Returns:
        tuple: (list of written variable names, dict of horizontal chunk sizes)
    """
    # Process and write each variable immediately to avoid accumulating all data in memory
    output_variables = []
    xy_chunks = {}  # horizontal chunk sizes, fixed by the first variable and shared by all variables and levels
    write_mode = mode
    for target_variable, var_options in input_config.get("variables", {}).items():
        ds = datasets.get(target_variable)
        if ds is None:
            continue
        source_variable = var_options.get('variable', target_variable)
        missing_value = var_options.get("missing_value", None)
        cast_option = var_options.get('mode', 'linear')
        int8_var = cast_to_int8(ds, source_variable, target_variable, cast_option, missing_value=missing_value,
                                data_min=var_options.get("data_min"), data_max=var_options.get("data_max"))
        if int8_var is not None:
            for dim in int8_var[target_variable].dims:
                if dim.startswith(("x", "y")) and dim not in xy_chunks:
                    xy_chunks[dim] = math.ceil(int8_var.sizes[dim] / horizontal_chunks)
            int8_var = int8_var.chunk(output_chunks(int8_var[target_variable], xy_chunks))
            int8_var.to_zarr(outfile, mode=write_mode, encoding={target_variable: VAR_ENCODING})
            output_variables.append(target_variable)
            write_mode = "a"
    return output_variables, xy_chunks


def write_coarsened_datasets(outfile, output_variables, xy_chunks, levels, coarsen):
    """Coarsen the int8 zarr dataset `levels` times and write each level to its own zarr store."""
    # Coarsen the dataset and write to zarr
    # NOTE: coarsening assumes all output variables share the same horizontal grid resolution
    ds = xr.open_zarr(outfile) if levels > 0 else None
    for level in range(1, levels + 1):
        coarsen_op = ds.coarsen({dim: 2 for dim in ds.dims if dim != "time"}, boundary="trim")
        ds = getattr(coarsen_op, coarsen)()
        level_outfile = outfile.replace(".zarr", f"-{level}.zarr")
        for var in output_variables:
            var_da = ds[var]
            var_da.chunk(output_chunks(var_da, xy_chunks)).to_zarr(
                level_outfile, mode="a", encoding={var: VAR_ENCODING})


def main(arg_list=None):
    """Convert the input dataset to int8 and save it in zarr format.

    Args:
        arg_list (list, optional): List of command-line arguments. Defaults to None, in which case sys.argv[1:] is used.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)
    args = parse_args(arg_list)

    if args.config is None:
        input_config = DALES_INPUT_CONFIG
    else:
        with open(args.config, "r") as f:
            input_config = yaml.safe_load(f)

    datasets = load_datasets(args, input_config)

    default_output_dir = args.input_dir if args.input_dir is not None else "."
    outfile = args.output if args.output is not None else os.path.join(default_output_dir, "output_int8.zarr")

    output_variables, xy_chunks = write_int8_dataset(datasets, input_config, outfile, args.mode,
                                                     args.horizontal_chunks)
    write_coarsened_datasets(input_config, outfile, output_variables, xy_chunks, args.levels, args.coarsen)


if __name__ == "__main__":
    main()
