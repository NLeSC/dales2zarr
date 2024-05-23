import argparse
import yaml
import xarray as xr
import zarr
from dales2zarr.zarr_cast import multi_cast_to_int8

# Parse command-line arguments
def parse_args(arg_list: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Convert input dataset to 8-bit integers and write to zarr")
    parser.add_argument("--input", metavar="FILE", type=str, required=True, help="Path to the input dataset file")
    parser.add_argument("--output", metavar="FILE", type=str, required=False, default=None, help="Path to the output zarr file")
    parser.add_argument("--config", metavar="FILE", type=str, required=False, default=None, help="Path to the input configuration file (yaml)")
    return parser.parse_args(args=arg_list)

def main(arg_list: list[str] | None = None):

    args = parse_args(arg_list)
        
    # Read the input dataset from file
    input_ds = xr.open_dataset(args.input)

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
