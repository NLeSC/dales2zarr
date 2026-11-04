"""Utilities for converting floating-point numerical data to unsigned 8-bit integers.

This module contains core functionalities to convert double-precision numerical data into
unsigned 8-bit integer values, suitable for visualization.
"""
import logging
import math
import dask
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


def normalize_data(data, data_min, data_max, mode, epsilon=1e-10):
    """Normalize input data array and cast to 8-bit integer.

    Args:
        data (array): Input numerical floating-point data to convert to 8-bit values
        data_min (float):  Global minimum of the input data. If data is a time slice of gridded values,
                           this should be the minimum over all spatial dimensions and time coordinate.
        data_max (float):  Global maximum of the input data. If data is a time slice of gridded values,
                           this should be the maximum over all spatial dimensions and time coordinate.
        mode (str): Mapping mode to short integers. Choices are `linear` or `log`. The latter is meant
                    for positive input data where small values need to be distinguishable.
        epsilon (float): Offset used for the logarithmic mapping.

    Returns:
        tuple (array, float, float): First value being the lower-precision array, second value is the
                                     min-value used for the final linear mapping to the interval [0,255],
                                     the last value is the max-value used for this mapping.

    Raises:
        ValueError: If `mode` is not `linear` or `log`.

    Example:
        This function can be called as follows:
        >>> from dales2zarr.my_module import normalize_data
        >>> normalize_data([1.0, 2.0, 3.0], 0.0, 10.0, 'linear')
        '[25, 50, 75]'
    """
    data = np.asarray(data)
    if mode == 'linear':
        scale_factor = 255.0 / (data_max - data_min)
        normalized_data = (data - data_min) * scale_factor
        scale_factors = data_min, data_max
    elif mode == 'log':
        log_min, log_max = np.log(data_min + epsilon), np.log(data_max + epsilon)
        normalized_data = np.log(data + epsilon)
        normalized_data -= log_min
        normalized_data *= 255.0 / (log_max - log_min)
        scale_factors = log_min, log_max
    else:
        raise ValueError(f"Mode {mode} is not supported: please use either linear or log")

    # Ensure the values are within the 8-bit integer range (0 to 255)
    return np.clip(normalized_data, 0, 255).astype('uint8'), *scale_factors


def prepare_data(input_ds, var_name, output_var=None, missing_value=None, data_min=None, data_max=None):
    """Select a variable, fill its missing values and determine its global range.

    This is done once, at the finest resolution. The result can be reused for all coarsened levels.
    Missing values (NaN or `missing_value`) are replaced by the global minimum.

    Args:
        input_ds (xarray.Dataset): The input dataset.
        var_name (str): The name of the variable to select.
        output_var (str, optional): Name of the returned array. Defaults to `var_name`.
        missing_value (float, optional): The value to be treated as missing in the input variable.
        data_min (float, optional): Global minimum. Computed from the data if not given.
        data_max (float, optional): Global maximum. Computed from the data if not given.

    Returns:
        tuple (xarray.DataArray, float, float): The filled data, the global minimum and the global maximum.
    """
    data = input_ds[var_name].rename(var_name if output_var is None else output_var)
    if missing_value is not None:
        data = data.where(data != float(missing_value))
    glob_min = data.min() if data_min is None else data_min
    glob_max = data.max() if data_max is None else data_max
    glob_min, glob_max = (float(v) for v in dask.compute(glob_min, glob_max))
    log.info(f'global min and max: {glob_min}, {glob_max}')
    return data.fillna(glob_min), glob_min, glob_max


def compute_z_range(data, data_min, data_max, mode='linear', epsilon=1e-10):
    """Determine the indices of the lowest and highest z layers that contain non-zero int8 values.

    Args:
        data (xarray.DataArray): Named input array with dimensions (time,) z, y, x, without missing values.
        data_min (float): Global minimum used for normalization.
        data_max (float): Global maximum used for normalization.
        mode (str, optional): The mode used for normalization. Default is 'linear'.
        epsilon (float, optional): Offset used for the logarithmic mapping.

    Returns:
        tuple (int, int): The bottom and top layer indices (kbot, ktop), or (-1, -1) if all layers are empty.
    """
    # Compute the maximum values for each horizontal layer
    layer_maxes = data.max(['x', 'y']).values
    log.info('layer maxes computed')

    # Normalize the layer maxes and get the indices of the bottom and top layers
    norms, _, _ = normalize_data(layer_maxes, data_min, data_max, mode, epsilon)
    nonzero_layers = np.flatnonzero(norms.reshape(-1, norms.shape[-1]).max(axis=0))
    kbot, ktop = (int(nonzero_layers[0]), int(nonzero_layers[-1])) if nonzero_layers.size else (-1, -1)
    log.info(f'computed kbot and ktop: {kbot}, {ktop}')
    return kbot, ktop


def cast_to_int8_3d(data, data_min, data_max, mode='linear', epsilon=1e-10, z_range=None):
    """Convert a 3d array (without missing values) to 8-bit integers, trimming empty z levels.

    Args:
        data (xarray.DataArray): Named input array with dimensions (time,) z, y, x.
        data_min (float): Global minimum used for normalization.
        data_max (float): Global maximum used for normalization.
        mode (str, optional): The mode used for normalization. Default is 'linear'.
        epsilon (float, optional): Offset used for the logarithmic mapping.
        z_range (tuple (int, int), optional): Bottom and top layer indices (kbot, ktop) to keep. Computed
                                              from the data with `compute_z_range` if not given.

    Returns:
        xarray.Dataset: A new dataset with the converted variable.

    Raises:
        ValueError: If `mode` is not `linear` or `log`.
    """
    var_name = data.name
    kbot, ktop = compute_z_range(data, data_min, data_max, mode, epsilon) if z_range is None else z_range

    # Get the heights associated with the bottom and top layers
    zbot, ztop = data.z[kbot].values, data.z[ktop].values
    log.info(f'computed associated heights are: {zbot}, {ztop}')

    # Create a new array with the appropriate shape
    new_shape = list(data.shape)
    z_index = 0 if len(new_shape) == 3 else 1
    new_shape[z_index] = ktop - kbot + 1
    arr = np.zeros(new_shape, dtype='uint8')

    # Convert each time slice to 8-bit integers
    if 'time' in data.dims:
        ntimes = data.sizes['time']
        for itime in range(ntimes):
            log.info(f'Converting time slice {itime} from {ntimes}...')
            arr[itime, ...], s1, s2 = normalize_data(data.isel(time=itime).isel(z=slice(kbot, ktop + 1)),
                                                     data_min, data_max, mode, epsilon)
    else:
        log.info("Converting single time slice...")
        arr[...], s1, s2 = normalize_data(data.isel(z=slice(kbot, ktop + 1)), data_min, data_max, mode, epsilon)

    # Create a new dataset with the converted variable
    output = data.isel(z=slice(kbot, ktop + 1)).to_dataset()
    output[var_name].values = arr
    output.attrs['lower_bound'] = s1
    output.attrs['upper_bound'] = s2
    output.attrs['zbot'] = zbot
    output.attrs['ztop'] = ztop

    return output


def cast_to_int8_2d(data, data_min, data_max, mode='linear', epsilon=1e-10):
    """Convert a 2D array (without missing values) to an 8-bit integer representation.

    Args:
        data (xarray.DataArray): Named input array.
        data_min (float): Global minimum used for normalization.
        data_max (float): Global maximum used for normalization.
        mode (str, optional): The mode used for normalization. Default is 'linear'.
        epsilon (float, optional): Offset used for the logarithmic mapping.

    Returns:
        xarray.Dataset: A new dataset with the converted variable.

    Raises:
        ValueError: If `mode` is not `linear` or `log`.
    """
    var_name = data.name
    # Create an array of zeros with the same shape as the input variable
    arr = np.zeros(list(data.shape), dtype='uint8')

    # Convert each time slice of the input variable to 8-bit integers
    if 'time' in data.dims:
        num_times = data.sizes['time']
        for itime in range(num_times):
            log.info(f'Converting time slice {itime} from {num_times}...')
            arr[itime, ...], s1, s2 = normalize_data(data.isel(time=itime), data_min, data_max, mode, epsilon)
    else:
        log.info('Converting single time slice...')
        arr[...], s1, s2 = normalize_data(data, data_min, data_max, mode, epsilon)

    # Create a new dataset with the converted variable
    output = data.to_dataset()
    output[var_name].values = arr
    output.attrs['lower_bound'] = s1
    output.attrs['upper_bound'] = s2
    return output


def cast_filled_to_int8(data, data_min, data_max, mode='linear', epsilon=1e-10, z_range=None):
    """Cast a prepared array (see `prepare_data`) to 8-bit integers, dispatching on its dimensionality.

    Args:
        z_range (tuple (int, int), optional): Bottom and top layer indices (kbot, ktop) to keep for 3d data,
                                              see `compute_z_range`. Computed from the data if not given.

    Returns:
        xarray.Dataset: A dataset with the casted variable; the z dimension is renamed to `z_<name>`.
    """
    if "z" not in data.dims:
        return cast_to_int8_2d(data, data_min, data_max, mode, epsilon)
    out = cast_to_int8_3d(data, data_min, data_max, mode, epsilon, z_range)
    # Trimmed z ranges differ per variable, so the z dimension and the height coordinates along it
    # must be renamed per variable
    z_name = "z_" + data.name
    renames = {c: f"{c}_{data.name}" for c in out.coords if out[c].dims == ("z",)}
    return out.rename_dims({"z": z_name}).rename_vars(renames)


def cast_to_int8(input_ds, input_var, output_var=None, mode='linear', epsilon=1e-10, missing_value=None,
                 data_min=None, data_max=None):
    """Casts a variable in the input dataset to an 8-bit integer.

    Args:
        input_ds (xarray.Dataset): The input dataset.
        input_var (str): The name of the variable to cast.
        output_var (str, optional): The name of the output variable.
                                    If not provided, the input variable name will be used.
        mode (str, optional): The casting mode. Defaults to 'linear'.
        epsilon (float, optional): Offset used for the logarithmic mapping.
        missing_value (float, optional): The value to be treated as missing in the input variable.
        data_min (float, optional): Global minimum used for normalization. Computed from the data if not given.
        data_max (float, optional): Global maximum used for normalization. Computed from the data if not given.

    Returns:
        xarray.Dataset: The input variable casted to an 8-bit integer, or None if it is not in the dataset.
    """
    if input_var not in input_ds.keys():
        log.warning(f'Variable {input_var} not found in the input dataset... skipping')
        return None
    data, data_min, data_max = prepare_data(input_ds, input_var, output_var, missing_value, data_min, data_max)
    return cast_filled_to_int8(data, data_min, data_max, mode, epsilon)
