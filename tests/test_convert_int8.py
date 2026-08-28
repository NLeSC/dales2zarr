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
    It creates per-variable input NetCDF files (using default filenames) and
    passes the directory to the main function. The test checks if the output
    zarr file exists and that the expected variables are present with the right dtype.

    Returns:
        None
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output.zarr")

        input_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ql_ds = xr.Dataset({'ql': (['zt', 'yt', 'xt'], input_data)})
        ql_ds.to_netcdf(os.path.join(temp_dir, "fielddump-ql.nc"))

        # Call the main function with the input directory
        main(["--input-dir", temp_dir, "--output", output_file])

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

    Creates per-variable input NetCDF files whose filenames are specified in the
    config, then verifies the merged output zarr contains both variables with the
    correct dtype and encoded values.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output.zarr")
        config_file = os.path.join(temp_dir, "config.yaml")

        ql_input_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        qr_input_data = np.array([[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]])

        xr.Dataset({'ql': (['zt', 'yt', 'xt'], ql_input_data)}).to_netcdf(
            os.path.join(temp_dir, "fielddump-ql.nc"))
        xr.Dataset({'qr': (['zt', 'yt', 'xt'], qr_input_data)}).to_netcdf(
            os.path.join(temp_dir, "fielddump-qr.nc"))

        input_config = {"ql": {"mode": "log"}, "qr": {"mode": "linear"}}

        with open(config_file, "w") as f:
            yaml.safe_dump(input_config, f)

        main(["--input-dir", temp_dir, "--output", output_file, "--config", config_file])

        assert os.path.exists(output_file)

        output_data = xr.open_zarr(output_file)

        assert "ql" in output_data
        assert "qr" in output_data

        assert output_data["ql"].dtype == "uint8"
        assert output_data["qr"].dtype == "uint8"

        assert output_data["ql"].values.flat[:3].tolist() == [0, 84, 134]
        assert output_data["qr"].values.flat[:3].tolist() == [0, 36, 72]


def test_main_coarsen_max():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output.zarr")
        lod1_file = os.path.join(temp_dir, "output-1.zarr")

        # 4x4x4 so coarsening by 2 yields a 2x2x2 level
        data = np.arange(64, dtype=float).reshape(4, 4, 4)
        ql_ds = xr.Dataset({"ql": (["zt", "yt", "xt"], data)})
        ql_ds.to_netcdf(os.path.join(temp_dir, "fielddump-ql.nc"))

        main(["--input-dir", temp_dir, "--output", output_file, "--levels", "1", "--coarsen", "max"])

        assert os.path.exists(lod1_file)
        lod1 = xr.open_zarr(lod1_file)
        assert "ql" in lod1
        assert lod1["ql"].dtype == "uint8"
        # max-pooling must produce values >= the mean-pooling equivalent
        mean_ds = xr.open_zarr(output_file)
        assert int(lod1["ql"].values.max()) >= int(mean_ds["ql"].values.max())


def test_main_custom_filename_in_config():
    """Test that a 'file' key in the config overrides the default input filename."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output.zarr")
        config_file = os.path.join(temp_dir, "config.yaml")

        input_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        xr.Dataset({'ql': (['zt', 'yt', 'xt'], input_data)}).to_netcdf(
            os.path.join(temp_dir, "custom-ql.nc"))

        input_config = {"ql": {"mode": "log", "file": "custom-ql.nc"}}
        with open(config_file, "w") as f:
            yaml.safe_dump(input_config, f)

        main(["--input-dir", temp_dir, "--output", output_file, "--config", config_file])

        output_data = xr.open_zarr(output_file)
        assert "ql" in output_data
        assert output_data["ql"].dtype == "uint8"


def test_main_with_thetavmix():
    """Test that thetavmix is loaded from cape-thetavmix.nc by default."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output.zarr")
        config_file = os.path.join(temp_dir, "config.yaml")

        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        xr.Dataset({'thetavmix': (['yt', 'xt'], input_data)}).to_netcdf(
            os.path.join(temp_dir, "cape-thetavmix.nc"))

        input_config = {"thetavmix": {"mode": "linear"}}
        with open(config_file, "w") as f:
            yaml.safe_dump(input_config, f)

        main(["--input-dir", temp_dir, "--output", output_file, "--config", config_file])

        output_data = xr.open_zarr(output_file)
        assert "thetavmix" in output_data
        assert output_data["thetavmix"].dtype == "uint8"
