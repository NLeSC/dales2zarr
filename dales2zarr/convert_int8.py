#!/usr/bin/env python

import argparse
import logging
import math
import os
import psutil
import xarray as xr
import zarr
import yaml
import numpy as np
from dales2zarr.zarr_cast import cast_filled_to_int8, compute_z_range, prepare_data

DALES_INPUT_CONFIG = {
    "variables": {
        "ql":       {"mode": "log",    "file": "fielddump-ql.nc", "variable": "ql"},
        "qr":       {"mode": "linear", "file": "fielddump-qr.nc", "variable": "sv002"},
        "thetavmix":{"mode": "linear", "file": "cape-thetavmix.nc", "variable": "thetavmix"},
    },
    "dimensions": {"x": "xt", "y": "yt", "z": "zt"}
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
                        help="Coarsening method for LOD levels, applied to the original data (so 'mean' is an "
                             "arithmetic mean of the data, i.e. a geometric mean in int8 space for log-mapped variables)")
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


def rename_dimensions(ds, dimensions):
    """Rename the input dimensions and their coordinates to the standard names x, y and z.

    Args:
        ds (xarray.Dataset): The input dataset.
        dimensions (dict): Maps standard dimension names (keys) to the dimension names in the input data (values).
    """
    renames = {src: dst for dst, src in dimensions.items() if src != dst and src in ds.dims}
    return ds.rename(renames)


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
                ds = xr.open_dataset(filepath, chunks='auto', engine=var_options.get("engine", None))
            if args.timestamps > 0:
                ds = ds.isel(time=slice(0, args.timestamps))
            datasets[target_variable] = rename_dimensions(ds, input_config.get("dimensions", {}))
        except FileNotFoundError:
            logging.warning(f'Input file {filepath} not found for variable {target_variable}... skipping')
    return datasets


def write_int8_dataset(datasets, input_config, outfile, mode, horizontal_chunks):
    """Cast each variable to int8 and write it to `outfile` (zarr).

    Returns:
        tuple: (written variable names, horizontal chunk sizes, per-variable (data_min, data_max) used for
        the cast, per-variable (kbot, ktop) z layer range (None for 2d variables), per-variable
        missing-value-filled data arrays to be used as base for coarsening)
    """
    # Process and write each variable immediately to avoid accumulating all data in memory
    output_variables = []
    data_ranges = {}
    z_ranges = {}
    filled_data = {}
    xy_chunks = {}  # horizontal chunk sizes, fixed by the first variable and shared by all variables and levels
    write_mode = mode
    for target_variable, var_options in input_config.get("variables", {}).items():
        ds = datasets.get(target_variable)
        if ds is None:
            continue
        source_variable = var_options.get('variable', target_variable)
        missing_value = var_options.get("missing_value", None)
        cast_option = var_options.get('mode', 'linear')
        if source_variable not in ds:
            logging.warning(f'Variable {source_variable} not found in the input dataset... skipping')
            continue
        logging.info(f'Processing variable {source_variable} as {target_variable}')
        data, data_min, data_max = prepare_data(ds, source_variable, target_variable, missing_value,
                                                var_options.get("data_min"), var_options.get("data_max"))
        z_range = compute_z_range(data, data_min, data_max, cast_option) if "z" in data.dims else None
        int8_var = cast_filled_to_int8(data, data_min, data_max, cast_option, z_range=z_range)
        filled_data[target_variable] = data
        z_ranges[target_variable] = z_range
        for dim in int8_var[target_variable].dims:
            if dim.startswith(("x", "y")) and dim not in xy_chunks:
                xy_chunks[dim] = math.ceil(int8_var.sizes[dim] / horizontal_chunks)
        int8_var = int8_var.chunk(output_chunks(int8_var[target_variable], xy_chunks))
        int8_var.to_zarr(outfile, mode=write_mode, encoding={target_variable: VAR_ENCODING})
        output_variables.append(target_variable)
        data_ranges[target_variable] = (data_min, data_max)
        write_mode = "a"
    return output_variables, xy_chunks, data_ranges, z_ranges, filled_data


# Default fraction of the currently available memory that a cached coarsened level may occupy; can be overridden
# with the environment variable named below.
CACHE_MEMORY_FRACTION = 0.5
CACHE_MEMORY_FRACTION_ENV = "DALES2ZARR_CACHE_MEMORY_FRACTION"


def cache_memory_fraction():
    """Return the memory fraction for caching, from the environment variable if set and valid."""
    value = os.environ.get(CACHE_MEMORY_FRACTION_ENV)
    if value is None:
        return CACHE_MEMORY_FRACTION
    try:
        fraction = float(value)
    except ValueError:
        fraction = None
    if fraction is None or not 0 <= fraction <= 1:
        logging.warning(f'Ignoring invalid {CACHE_MEMORY_FRACTION_ENV}={value!r} (expected a number in [0, 1]), '
                        f'using {CACHE_MEMORY_FRACTION}')
        return CACHE_MEMORY_FRACTION
    return fraction


def fits_in_memory(da, fraction=None):
    """Return True if `da` can be loaded eagerly using at most `fraction` of the available memory.

    `fraction` defaults to the value of the environment variable DALES2ZARR_CACHE_MEMORY_FRACTION, or 0.5.
    """
    if fraction is None:
        fraction = cache_memory_fraction()
    available = psutil.virtual_memory().available
    fits = da.nbytes <= fraction * available
    if not fits:
        logging.info(f'Not caching {da.name}: needs {da.nbytes / 2**20:.0f} MiB, '
                     f'{available / 2**20:.0f} MiB available (fraction {fraction})')
    return fits


def write_coarsened_datasets(filled_data, input_config, outfile, output_variables, xy_chunks, data_ranges,
                             z_ranges, levels, coarsen):
    """Coarsen the input data `levels` times, cast it to int8 and write each level to its own zarr store.

    Coarsening is done on the missing-value-filled original data returned by `write_int8_dataset`, so `mean`
    is an arithmetic mean of the data. Every level is cast with the data range and the bottom and top z layers
    of the full-resolution variable, so all levels share the same mapping to int8. Only the horizontal
    dimensions are coarsened. A coarsened level that is the input of the next level is loaded into memory
    (cached) if enough memory is available, otherwise it stays lazy and is recomputed when needed.
    """
    # NOTE: coarsening assumes all output variables share the same horizontal grid resolution
    for var in output_variables:
        var_options = input_config["variables"][var]
        data_min, data_max = data_ranges[var]
        z_range = z_ranges[var]
        da = filled_data[var]
        for level in range(1, levels + 1):
            logging.info(f'Coarsening variable {var} to level {level}')
            coarsen_op = da.coarsen({dim: 2 for dim in da.dims if dim.startswith(("x", "y"))}, boundary="trim")
            da = getattr(coarsen_op, coarsen)()
            if level < levels and fits_in_memory(da):
                da = da.load()
            int8_var = cast_filled_to_int8(da, data_min, data_max, var_options.get("mode", "linear"),
                                           z_range=z_range)
            int8_var = int8_var.chunk(output_chunks(int8_var[var], xy_chunks))
            level_outfile = outfile.replace(".zarr", f"-{level}.zarr")
            int8_var.to_zarr(level_outfile, mode="a", encoding={var: VAR_ENCODING})


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

    output_variables, xy_chunks, data_ranges, z_ranges, filled_data = write_int8_dataset(
        datasets, input_config, outfile, args.mode, args.horizontal_chunks)
    write_coarsened_datasets(filled_data, input_config, outfile, output_variables, xy_chunks, data_ranges,
                             z_ranges, args.levels, args.coarsen)


if __name__ == "__main__":
    main()
