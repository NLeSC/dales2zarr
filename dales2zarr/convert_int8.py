import argparse
import xarray as xr
import yaml
import zarr
import logging
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

    # Call multi_cast_to_int8 on the input dataset
    output_ds, output_variables = multi_cast_to_int8(input_ds, input_config)

    outfile = args.output if args.output is not None else args.input.replace(".nc", "_int8.zarr")

    # Write the result to zarr with Blosc compression
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    var_encoding = {"dtype": "uint8", "compressor": compressor}
    output_ds.to_zarr(outfile, mode="w", encoding={var: var_encoding for var in output_variables})

if __name__ == "__main__":
    main()
