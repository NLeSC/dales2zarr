import os
import tempfile
import numpy as np
import xarray as xr
import yaml
from dales2zarr.convert_int8 import main

# These tests have been created with the help of github copilot

def test_main_with_default_config():
    """Test the main function with the default configuration.

    This test case creates a temporary directory to store the output zarr file.
    It sets the input and output file paths and creates a sample input dataset.
    The input dataset is saved to a netCDF file and then passed to the main function.
    The test checks if the output zarr file exists and reads the output dataset from it.
    It also checks if the output dataset has the expected variables and data types.

    Returns:
        None
    """
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
    """Test the main function with a custom configuration.

    This test case performs the following steps:
    1. Creates a temporary directory to store the output zarr file.
    2. Sets the input and output file paths.
    3. Creates a sample input dataset with ql and qr variables.
    4. Saves the input dataset to a netCDF file.
    5. Creates a sample input configuration.
    6. Saves the input configuration to a yaml file.
    7. Calls the main function with the input, output, and config file paths.
    8. Checks if the output zarr file exists.
    9. Reads the output dataset from the zarr file.
    10. Checks if the output dataset has the expected variables.
    11. Checks if the output dataset variables have the expected data type.
    12. Checks if the output dataset variables have the expected values.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, "input.nc")
        output_file = os.path.join(temp_dir, "output.zarr")
        config_file = os.path.join(temp_dir, "config.yaml")

        ql_input_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        qr_input_data = np.array([[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]])
        input_ds = xr.Dataset({'ql': (['zt', 'yt', 'xt'], ql_input_data), 'qr': (['zt', 'yt', 'xt'], qr_input_data)})

        input_ds.to_netcdf(input_file)

        input_config = {"ql": {"mode": "log"}, "qr": {"mode": "linear"}}

        with open(config_file, "w") as f:
            yaml.safe_dump(input_config, f)

        main(["--input", input_file, "--output", output_file, "--config", config_file])

        assert os.path.exists(output_file)

        output_data = xr.open_zarr(output_file)

        assert "ql" in output_data
        assert "qr" in output_data

        assert output_data["ql"].dtype == "uint8"
        assert output_data["qr"].dtype == "uint8"

        assert output_data["ql"].values.flat[:3].tolist() == [0, 84, 134]
        assert output_data["qr"].values.flat[:3].tolist() == [0, 36, 72]
