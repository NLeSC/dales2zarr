"""Utilities for converting floating-point numerical data to unsigned 8-bit integers.

This module contains core functionalities to convert double-precision numerical data into
unsigned 8-bit integer values, suitable for visualization.
"""
import logging
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
    if mode == 'linear':
        # Calculate the scale factor and offset to map the data to the 8-bit integer range (0 to 255)
        scale_factor = 255.0 / (data_max - data_min)
        offset = -data_min * scale_factor
        scale_factors = data_min, data_max

        # Apply the scale factor and offset to normalize the data
        normalized_data = scale_factor * data + offset

    elif mode == 'log':
        # Apply the logarithmic transformation to highlight small values
        log_data, log_min, log_max = np.log(data + epsilon), np.log(data_min + epsilon), np.log(data_max + epsilon)

        # Scale the data to the 8-bit integer range (0 to 255)
        normalized_data = 255.0 * (log_data - log_min) / (log_max - log_min)
        scale_factors = log_min, log_max
    else:
        raise ValueError(f"Mode {mode} is not supported: please use either linear or log")

    # Ensure the values are within the 8-bit integer range (0 to 255)
    return np.clip(normalized_data, 0, 255).astype('uint8'), *scale_factors


def cast_to_int8_3d(input_ds, var_name, mode='linear', epsilon=1e-10):
    """Convert 3d xarray to 8-bit integers.

    Args:
        input_ds (xarray.Dataset): Input xarray dataset
        var_name (str): The name of the variable to be converted.
        mode (str, optional): The mode used for normalization. Default is 'linear'.
        epsilon (float, optional): Offset used for the logarithmic mapping.

    Returns: (xarray.Dataset)
        xarray.Dataset: A new dataset with the converted variable.

    Raises:
        ValueError: If `mode` is not `linear` or `log`.

    Example:
        This function can be called as follows:
        >>> from dales2zarr.my_module import cast_to_8bit_integer_3d
        >>> cast_to_8bit_integer_3d([1.0, 2.0, 3.0], 0.0, 10.0, 'linear')
        '[25, 50, 75]'
    """
    # Get the input data and compute the global minimum
    input_data = input_ds[var_name]
    glob_min = input_data.min().values
    log.info(f'global minimum is {glob_min}')

    # Compute the maximum values for each layer
    layer_maxes = input_data.max(['xt', 'yt']).values
    log.info(f'layer maxes are {layer_maxes}')

    # Compute the global maximum
    glob_max = np.max(layer_maxes)
    log.info(f'computed min and max: {glob_min}, {glob_max}')

    # Normalize the layer maxes and get the indices of the bottom and top layers
    norms, _, _ = normalize_data(layer_maxes, glob_min, glob_max, mode, epsilon)
    kbot, ktop, kmax = -1, -1, norms.shape[-1]
    for k in range(kmax):
        if np.max(norms[..., k]) != 0 and kbot < 0:
            kbot = k
        if np.max(norms[..., kmax - k - 1]) != 0 and ktop < 0:
            ktop = kmax - k - 1
    log.info(f'computed kbot and ktop: {kbot}, {ktop}')

    # Get the heights associated with the bottom and top layers
    zbot, ztop = input_ds.zt[kbot], input_ds.zt[ktop]
    log.info(f'computed associated heights are: {zbot}, {ztop}')

    # Create a new array with the appropriate shape
    new_shape = list(input_data.shape)
    z_index = 0 if len(new_shape) == 3 else 1
    new_shape[z_index] = ktop - kbot + 1
    arr = np.zeros(new_shape, dtype='uint8')

    # Convert each time slice to 8-bit integers
    if 'time' in input_ds.sizes:
        ntimes = input_ds.sizes['time']
        for itime in range(ntimes):
            log.info(f'Converting time slice {itime} from {ntimes}...')
            arr[itime, ...], s1, s2 = normalize_data(input_data.isel(time=itime).isel(zt=slice(kbot, ktop + 1)),
                                                     glob_min, glob_max, mode, epsilon)
    else:
        log.info("Converting single time slice...")
        arr[...], s1, s2 = normalize_data(input_data.isel(zt=slice(kbot, ktop + 1)),
                                          glob_min, glob_max, mode, epsilon)

    # Create a new dataset with the converted variable
    output = input_data.isel(zt=slice(kbot, ktop + 1)).to_dataset()
    output[var_name].values = arr
    output.attrs['lower_bound'] = s1
    output.attrs['upper_bound'] = s2
    output.attrs['zbot'] = zbot
    output.attrs['ztop'] = ztop

    return output


def cast_to_int8_2d(input_ds, var_name, mode='linear', epsilon=1e-10):
    """Converts a 2D variable in a given xarray Dataset to an 8-bit integer representation.

    Args:
        input_ds (xarray.Dataset): The input dataset containing the variable to be converted.
        var_name (str): The name of the variable to be converted.
        mode (str, optional): The mode used for normalization. Default is 'linear'.
        epsilon (float, optional): Offset used for the logarithmic mapping.

    Returns: (xarray.Dataset)
        xarray.Dataset: A new dataset with the converted variable.

    Raises:
        ValueError: If `mode` is not `linear` or `log`.

    Examples:
        # Convert a 2D variable named 'temperature' in the input dataset to an 8-bit integer representation:
        output_ds = cast_to_8bit_integer_2d(input_ds, 'temperature')

        # Convert a 2D variable named 'humidity' in the input dataset to an 8-bit integer representation:
        # using 'log' mode:
        output_ds = cast_to_8bit_integer_2d(input_ds, 'humidity', mode='log')
    """
    # Compute the global minimum and maximum values of the input variable
    glob_min = input_ds[var_name].min().values
    glob_max = input_ds[var_name].max().values
    log.info(f'computed min and max: {glob_min}, {glob_max}')

    # Create an array of zeros with the same shape as the input variable
    arr = np.zeros(list(input_ds[var_name].shape), dtype='uint8')

    # Convert each time slice of the input variable to 8-bit integers
    if 'time' in input_ds.sizes:
        num_times = input_ds.sizes['time']
        for itime in range(num_times):
            log.info(f'Converting time slice {itime} from {num_times}...')
            arr[itime,...], s1, s2 =  normalize_data(input_ds[var_name].isel(time=itime),
                                                     glob_min, glob_max, mode, epsilon)
    else:
        log.info('Converting single time slice...')
        arr[...], s1, s2 =  normalize_data(input_ds[var_name], glob_min, glob_max, mode, epsilon)

    # Create a new dataset with the converted variable
    output = input_ds[var_name].to_dataset()
    output[var_name].values = arr
    output.attrs['lower_bound'] = s1
    output.attrs['upper_bound'] = s2
    return output


def cast_to_int8(input_ds, input_var, output_var=None, mode='linear', epsilon=1e-10):
    """Casts a variable in the input dataset to an 8-bit integer.

    Args:
        input_ds (xarray.Dataset): The input dataset.
        input_var (str): The name of the variable to cast.
        output_var (str, optional): The name of the output variable.
                                    If not provided, the input variable name will be used.
        mode (str, optional): The casting mode. Defaults to 'linear'.
        epsilon (float, optional): Offset used for the logarithmic mapping.

    Returns:
        xarray.Dataset: The input dataset with the variable casted to an 8-bit integer.
    """
    if input_var not in input_ds.keys():
        log.warning(f'Variable {input_var} not found in the input dataset... skipping')
        return None
    var_name = input_var if output_var is None else output_var
    if var_name != input_var:
        log.info(f'Converting variable {input_var} to {var_name}...')
        input_ds = input_ds.rename_vars({input_var: var_name})
    dims = input_ds[var_name].coords
    if "zt" in dims or "zm" in dims:
        return (cast_to_int8_3d(input_ds, var_name, mode, epsilon)
                .rename_dims({"zt": "z_" + var_name})
                .rename_vars({"zt": "z_" + var_name}))
    else:
        return cast_to_int8_2d(input_ds, var_name, mode, epsilon)


def multi_cast_to_int8(input_ds, input_config):
    """Casts multiple variables in the input dataset to 8-bit integers.

    Args:
        input_ds (xarray.Dataset): The input dataset.
        input_config (dict): A dictionary containing the configuration for casting the variables.

    Returns:
        xarray.Dataset: The input dataset with the variables casted to 8-bit integers.
    """
    outputs, variables = [], []
    for input_var, var_options in input_config.items():
        int8_var = cast_to_int8(input_ds, input_var, **var_options)
        if int8_var is not None:
            outputs.append(int8_var)
            variables.append(var_options.get('output_var', input_var))
    return xr.merge(outputs), variables
