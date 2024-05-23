"""Tests for the dales2zarr.my_module module."""
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from dales2zarr.zarr_cast import normalize_data, cast_to_int8_2d, cast_to_int8_3d, multi_cast_to_int8

# These tests have been created with the help of github copilot

def test_normalize_data_linear():
    data = np.array([1.0, 2.0, 3.0])
    data_min = 0.0
    data_max = 10.0
    mode = 'linear'
    expected_result = np.array([25, 51, 76], dtype='uint8')
    result, _, _ = normalize_data(data, data_min, data_max, mode)
    assert np.array_equal(result, expected_result)


def test_normalize_data_log():
    data = np.array([1.0, 2.0, 3.0])
    data_min = 0.0
    data_max = 10.0
    mode = 'log'
    expected_result = np.array([231, 238, 242], dtype='uint8')
    result, _, _ = normalize_data(data, data_min, data_max, mode)
    assert np.array_equal(result, expected_result)


def test_normalize_data_invalid_mode():
    data = np.array([1.0, 2.0, 3.0])
    data_min = 0.0
    data_max = 10.0
    mode = 'invalid'
    with pytest.raises(ValueError):
        normalize_data(data, data_min, data_max, mode)


def test_cast_to_int8():
    # Create a sample input dataset
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_ds = xr.Dataset({'temperature': (['yt', 'xt'], input_data)})

    # Call the function to convert the variable to 8-bit integers
    output_ds = cast_to_int8_2d(input_ds, 'temperature')

    # Check the output dataset
    expected_data = np.array([[0, 51, 102], [153, 204, 255]], dtype='uint8')
    assert np.array_equal(output_ds['temperature'].values, expected_data)


def test_cast_to_int8_2d_attrs():
    # Create a sample input dataset
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_ds = xr.Dataset({'temperature': (['yt', 'xt'], input_data)})

    # Call the function to convert the variable to 8-bit integers
    output_ds = cast_to_int8_2d(input_ds, 'temperature')

    # Check the output dataset
    expected_data = np.array([[0, 51, 102], [153, 204, 255]], dtype='uint8')
    assert output_ds.attrs['lower_bound'] == 1.0
    assert output_ds.attrs['upper_bound'] == 6.0


def test_cast_to_int8_2d_time_dep():
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