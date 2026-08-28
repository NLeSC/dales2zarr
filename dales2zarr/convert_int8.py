#!/usr/bin/env python

import argparse
import logging
import xarray as xr
import zarr
import yaml
from dales2zarr.zarr_cast import multi_cast_to_int8

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
    parser.add_argument("--input", metavar="FILE", type=str, required=True,
                        help="Path to the input dataset file")
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

    # Read the input dataset from file
    input_ds = xr.open_dataset(args.input, chunks='auto')

    if args.config is None:
        # Default input configurationz
        input_config = {"ql": {"mode": "log"}, "qr": {"mode": "linear"}}
    else:
        # Read the input configuration from yaml
        with open(args.config, "r") as f:
            input_config = yaml.safe_load(f)

    # Keep only the first args.timestamps timestamps
    if args.timestamps > 0:
        input_ds = input_ds.isel(time=slice(0, args.timestamps))

    # Call multi_cast_to_int8 on the input dataset
    output_ds, output_variables = multi_cast_to_int8(input_ds, input_config)

    outfile = args.output if args.output is not None else args.input.replace(".nc", "_int8.zarr")

    # Write the result to zarr with Blosc compression
    compressor = zarr.Blosc(cname="lz4", clevel=6, shuffle=zarr.Blosc.BITSHUFFLE)
    var_encoding = {"dtype": "uint8", "compressor": compressor}
    output_ds.to_zarr(outfile, mode=args.mode, encoding={var: var_encoding for var in output_variables})

    # Coarsen the dataset and write to zarr
    ds = output_ds
    for level in range(1, args.levels + 1):
        coarsen_op = ds.coarsen({dim: 2 for dim in ds.dims if dim != "time"}, boundary="trim")
        ds = getattr(coarsen_op, args.coarsen)()
        ds.to_zarr(outfile.replace(".zarr", f"-{level}.zarr"), mode="a", encoding={var: var_encoding for var in output_variables})


if __name__ == "__main__":
    main()
