"""Tests for the dales2zarr.my_module module."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dales2zarr.zarr_cast import cast_to_int8
from dales2zarr.zarr_cast import cast_to_int8_2d
from dales2zarr.zarr_cast import cast_to_int8_3d
from dales2zarr.zarr_cast import multi_cast_to_int8
from dales2zarr.zarr_cast import normalize_data

# These tests have been created with the help of github copilot

def test_normalize_data_linear():
    """Test case for the `normalize_data` function with linear mode.

    This test checks if the `normalize_data` function correctly normalizes the input data using
    linear mode. It creates a numpy array `data` with values [1.0, 2.0, 3.0], and sets the data_min
    and data_max values to 0.0 and 10.0 respectively. The expected result is a numpy array
    [25, 51, 76] with dtype 'uint8'.
    The function then calls the `normalize_data` function with the given input and checks if the
    result matches the expected result.

    Raises:
        AssertionError: If the result of `normalize_data` does not match the expected result.
    """
    data = np.array([1.0, 2.0, 3.0])
    data_min = 0.0
    data_max = 10.0
    mode = 'linear'
    expected_result = np.array([25, 51, 76], dtype='uint8')
    result, _, _ = normalize_data(data, data_min, data_max, mode)
    assert np.array_equal(result, expected_result)


def test_normalize_data_log():
    """Test case for the `normalize_data` function with logarithmic mode.

    This test checks if the `normalize_data` function correctly normalizes the input data using
    log mode. It creates a numpy array `data` with values [1.0, 2.0, 3.0], and sets the data_min
    and data_max values to 0.0 and 10.0 respectively. The expected result is a numpy array
    [231, 238, 242] with dtype 'uint8'.
    The function then calls the `normalize_data` function with the given input and checks if the
    result matches the expected result.

    Raises:
        AssertionError: If the result of `normalize_data` does not match the expected result.
    """
    data = np.array([1.0, 2.0, 3.0])
    data_min = 0.0
    data_max = 10.0
    mode = 'log'
    expected_result = np.array([231, 238, 242], dtype='uint8')
    result, _, _ = normalize_data(data, data_min, data_max, mode)
    assert np.array_equal(result, expected_result)


def test_normalize_data_invalid_mode():
    """Test case for the `normalize_data` function when an invalid mode is provided.

    This test checks if the `normalize_data` function raises a `ValueError` when an invalid mode is provided.
    The function takes an input data array, minimum and maximum values, and a mode as arguments.
    It expects the `ValueError` to be raised when an invalid mode is provided.

    Returns:
    - None
    """
    data = np.array([1.0, 2.0, 3.0])
    data_min = 0.0
    data_max = 10.0
    mode = 'invalid'
    with pytest.raises(ValueError):
        normalize_data(data, data_min, data_max, mode)


def test_cast_to_int8():
    """Test function for casting a variable to 8-bit integers.

    This function creates a sample input dataset with temperature values and calls the `cast_to_int8_2d` function
    to convert the temperature variable to 8-bit integers. It then checks if the output dataset matches the expected
    data.

    Returns:
        None
    """
    # Create a sample input dataset
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_ds = xr.Dataset({'temperature': (['yt', 'xt'], input_data)})

    # Call the function to convert the variable to 8-bit integers
    output_ds = cast_to_int8_2d(input_ds, 'temperature')

    # Check the output dataset
    expected_data = np.array([[0, 51, 102], [153, 204, 255]], dtype='uint8')
    assert np.array_equal(output_ds['temperature'].values, expected_data)


def test_unavailable_variable():
    """Test function for casting a variable to 8-bit integers.

    This function creates a sample input dataset with temperature values and calls the `cast_to_int8_2d` function
    to convert the temperature variable to 8-bit integers. It then checks if the output dataset matches the expected
    data.

    Returns:
        None
    """
    # Create a sample input dataset
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_ds = xr.Dataset({'temperature': (['yt', 'xt'], input_data)})

    # Call the function to convert the variable to 8-bit integers
    output_ds = cast_to_int8(input_ds, 'rain')

    # Check the output dataset
    assert output_ds is None


def test_cast_to_int8_2d_attrs():
    """Test function for casting a 2D variable to 8-bit integers and checking the output dataset attributes.

    This function creates a sample input dataset with a variable named 'temperature' and calls the 'cast_to_int8_2d'
    function to convert the variable to 8-bit integers. It then checks the output dataset attributes to ensure that
    the lower and upper bounds are correctly set.

    Returns:
        None
    """
    # Create a sample input dataset
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_ds = xr.Dataset({'temperature': (['yt', 'xt'], input_data)})

    # Call the function to convert the variable to 8-bit integers
    output_ds = cast_to_int8_2d(input_ds, 'temperature')

    # Check the output dataset
    assert output_ds.attrs['lower_bound'] == 1.0
    assert output_ds.attrs['upper_bound'] == 6.0


def test_cast_to_int8_2d_time_dep():
    """Test function for casting a 2D variable with a time dimension to 8-bit integers.

    This function creates a sample input dataset with a time dimension and a variable called 'temperature'.
    It then calls the 'cast_to_int8_2d' function to convert the 'temperature' variable to 8-bit integers.
    The output dataset is checked against the expected data.

    Returns:
        None
    """
    # Create a sample input dataset with time dimension
    input_data = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
    time = pd.date_range('2022-01-01', periods=2)
    input_ds = xr.Dataset({'temperature': (['time', 'yt', 'xt'], input_data)},
                          coords={'time': time})

    # Call the function to convert the variable to 8-bit integers
    output_ds = cast_to_int8_2d(input_ds, 'temperature')

    # Check the output dataset
    expected_data = np.array([[[0, 23, 46], [69, 92, 115]],
                              [[139, 162, 185], [208, 231, 255]]], dtype='uint8')
    assert np.array_equal(output_ds['temperature'].values, expected_data)


def test_cast_to_int8_2d_time_dep_attrs():
    """Test case for casting a 2D variable with time-dependent attributes to int8.

    This test case creates a sample input dataset with a time dimension and calls
    the `cast_to_int8_2d` function to convert the 'temperature' variable to 8-bit integers.
    It then asserts that the lower and upper bounds of the output dataset's attributes
    match the expected values.

    Returns:
        None
    """
    # Create a sample input dataset with time dimension
    input_data = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
    time = pd.date_range('2022-01-01', periods=2)
    input_ds = xr.Dataset({'temperature': (['time', 'yt', 'xt'], input_data)},
                          coords={'time': time})

    # Call the function to convert the variable to 8-bit integers
    output_ds = cast_to_int8_2d(input_ds, 'temperature')

    assert output_ds.attrs['lower_bound'] == 1.0
    assert output_ds.attrs['upper_bound'] == 12.0


def test_cast_to_int8_3d_time_dep():
    """Test function for casting a 3D time-dependent variable to 8-bit integers.

    This function creates a sample input dataset with a 3D time-dependent variable called 'temperature'.
    It then calls the 'cast_to_int8_3d' function to convert the 'temperature' variable to 8-bit integers.
    The function checks the output dataset to ensure that the conversion was done correctly.

    Returns:
        None
    """
    # Create a sample input dataset
    input_data = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[0.0, 0.0], [0.0, 0.0]]],
                           [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    time = pd.date_range('2022-01-01', periods=2)
    input_ds = xr.Dataset({'temperature': (['time', 'zt', 'yt', 'xt'], input_data)},
                          coords={'time': time, 'zt': [0.0, 1500.0, 3200.0], 'yt': [0, 1], 'xt': [0, 1]})

    # Call the function to convert the variable to 8-bit integers
    output_ds = cast_to_int8_3d(input_ds, 'temperature')

    # Check the output dataset
    expected_data = np.array([[[[15, 31], [47, 63]], [[79, 95], [111, 127]]],
                           [[[143, 159], [175, 191]], [[207, 223], [239, 255]]]], dtype='uint8')
    assert np.array_equal(output_ds['temperature'].values, expected_data)
    assert output_ds.attrs['ztop'] == 1500.0
    assert output_ds.attrs['zbot'] == 0.0


def test_multi_cast_to_int8():
    """Test function for multi_cast_to_int8.

    This function tests the functionality of the multi_cast_to_int8 function by creating a sample input dataset,
    defining the input configuration, calling the function to convert the variables to 8-bit integers, and checking
    the output dataset.

    Returns:
        None
    """
    # Create a sample input dataset
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_ds = xr.Dataset({'temperature': (['yt', 'xt'], input_data),
                           'humidity': (['yt', 'xt'], input_data)})

    # Define the input configuration
    input_config = {'temperature': {'mode': 'linear'},
                    'humidity': {'mode': 'linear', 'output_var': 'qt'}}

    # Call the function to convert the variables to 8-bit integers
    output_ds,_ = multi_cast_to_int8(input_ds, input_config)

    # Check the output dataset
    expected_data = np.array([[0, 51, 102], [153, 204, 255]], dtype='uint8')
    assert np.array_equal(output_ds['temperature'].values, expected_data)
    assert np.array_equal(output_ds['qt'].values, expected_data)
