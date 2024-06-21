import mlx.core as mx

def custom_roll(arr, shift, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    if not isinstance(shift, tuple):
        shift = (shift,)
    
    for ax, sh in zip(axis, shift):
        arr = custom_roll_single_axis(arr, sh, ax)
    
    return arr

def custom_roll_single_axis(arr, shift, axis):
    if shift == 0:
        return arr
    
    shape = arr.shape
    n = shape[axis]
    
    shift = shift % n 
    
    indices = mx.concatenate((mx.arange(n - shift, n), mx.arange(n - shift)))
    
    return mx.take(arr, indices, axis=axis)


def window_partition(input_feature, window_size):
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.reshape(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, num_channels)
    return windows


def window_reverse(windows, window_size, height, width):
    num_channels = windows.shape[-1]
    windows = windows.reshape(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, height, width, num_channels)
    return windows