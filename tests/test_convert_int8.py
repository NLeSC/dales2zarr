import os
import tempfile
import numpy as np
import yaml
import xarray as xr
from dales2zarr.convert_int8 import main

# These tests have been created with the help of github copilot

def test_main_with_default_config():
    # Create a temporary directory to store the output zarr file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set the input and output file paths
        input_file = os.path.join(temp_dir, "input.nc")
        output_file = os.path.join(temp_dir, "output.zarr")

        input_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        input_ds = xr.Dataset({'ql': (['zt', 'yt', 'xt'], input_data)})

        # Save the input dataset to a netCDF file
        input_ds.to_netcdf(input_file)

        # Call the main function
        main(["--input", input_file, "--output", output_file])

        # Check if the output zarr file exists
        assert os.path.exists(output_file)

        # Read the output dataset from the zarr file
        output_data = xr.open_zarr(output_file)

        # Check if the output dataset has the expected variables
        assert "ql" in output_data
        assert "qr" not in output_data

        # Check if the output dataset variables have the expected data type
        assert output_data["ql"].dtype == "uint8"


def test_main_with_custom_config():
    # Create a temporary directory to store the output zarr file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set the input and output file paths
        input_file = os.path.join(temp_dir, "input.nc")
        output_file = os.path.join(temp_dir, "output.zarr")
        config_file = os.path.join(temp_dir, "config.yaml")

        # Create a sample input dataset
        ql_input_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        qr_input_data = np.array([[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]])
        input_ds = xr.Dataset({'ql': (['zt', 'yt', 'xt'], ql_input_data), 'qr': (['zt', 'yt', 'xt'], qr_input_data)})

        # Save the input dataset to a netCDF file
        input_ds.to_netcdf(input_file)

        # Create a sample input configuration
        input_config = {"ql": {"mode": "log"}, "qr": {"mode": "linear"}}

        # Save the input configuration to a yaml file
        with open(config_file, "w") as f:
            yaml.safe_dump(input_config, f)

        # Call the main function
        main(["--input", input_file, "--output", output_file, "--config", config_file])

        # Check if the output zarr file exists
        assert os.path.exists(output_file)

        # Read the output dataset from the zarr file
        output_data = xr.open_zarr(output_file)

        # Check if the output dataset has the expected variables
        assert "ql" in output_data
        assert "qr" in output_data

        # Check if the output dataset variables have the expected data type
        assert output_data["ql"].dtype == "uint8"
        assert output_data["qr"].dtype == "uint8"

        # Check if the output dataset variables have the expected values
        assert output_data["ql"].values.flat[:3].tolist() == [0, 84, 134]
        assert output_data["qr"].values.flat[:3].tolist() == [0, 36, 72]