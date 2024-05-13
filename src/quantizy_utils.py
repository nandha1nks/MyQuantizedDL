from typing import Union
import torch


def get_scale_and_zero_point(data: torch.Tensor, dtype: torch.dtype,
                             symmetric: bool = True, mode: str = "min_max",
                             dim: Union[None, int] = None) -> (torch.Tensor, torch.Tensor):
    """
    Given data tensor returns the scale and zero point for the quantized tensor.

    :param data: Input data tensor
    :param dtype: Quantized Vector's dtype
    :param symmetric: Whether the quantized vector space will be symmetric on 0
    :param mode: Method to find the min and max value of the data
    :param dim: Dimension to define per tensor (dim=None) or per channel (dim=dim_index)
    :return: two tensors scale and zero point which can be used to optimized for tensor
    """
    if symmetric:
        yq = torch.tensor(torch.iinfo(dtype).max).to(dtype)
        xq = -yq
    else:
        yq = torch.tensor(torch.iinfo(dtype).max).to(dtype)
        xq = torch.tensor(torch.iinfo(dtype).min).to(dtype)

    if dim is None:
        x, y = get_alpha_beta(data, mode=mode)
    else:
        x = torch.zeros(data.size(dim))
        y = torch.zeros(data.size(dim))
        for idx in range(data.size(dim)):
            x[idx], y[idx] = get_alpha_beta(data.select(dim, idx), mode=mode)

        reshape_size = [1] * data.dim()
        reshape_size[dim] = -1
        y = y.reshape(reshape_size)
        x = x.reshape(reshape_size)

    if symmetric:
        y = torch.maximum(x.abs(), y.abs())
        x = -y

    scale = (y - x) / (yq.float() - xq.float())

    if symmetric:
        zero_point = torch.zeros_like(scale)
    else:
        zero_point = (y * xq - x * yq) / (y - x)

    if dim is not None:
        zero_point[zero_point < xq] = xq
        zero_point[zero_point > yq] = yq
        zero_point = torch.round(zero_point).to(dtype)
    else:
        if zero_point < xq:
            zero_point = torch.tensor(xq, dtype=dtype)
        elif zero_point > yq:
            zero_point = torch.tensor(yq, dtype=dtype)
        else:
            zero_point = torch.round(zero_point).to(dtype)

    return scale, zero_point


def get_alpha_beta(x: torch.Tensor, mode: str = "min_max") -> (torch.Tensor, torch.Tensor):
    """
    Get the range of the tensor based on the mode.

    :param x: data tensor
    :param mode: One of the min_max
    :return: tensor representing min and tensor representing max
    """
    if mode == "min_max":
        return x.min(), x.max()
    raise NotImplementedError()


def quantize_with_scale_and_zero_point(x: torch.Tensor, scale: torch.Tensor,
                                       zero_point: torch.Tensor,
                                       dtype: torch.dtype = torch.int8):
    """
    Quantize the data vector with scale and zero point.

    :param x: data tensor
    :param scale: tensor representing scale
    :param zero_point: tensor representing zero point
    :param dtype: quantized vectors dtype
    :return: quantized tensor
    """
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    return torch.clip(torch.round(x / scale + zero_point), q_min, q_max).to(dtype)


def dequantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    Dequantize the quantized vector using scale and zero point.

    :param x: data tensor
    :param scale: tensor representing scale
    :param zero_point: tensor representing zero point
    :return: Dequantized vector
    """
    return scale * (x.float() - zero_point)


def quantize_int(x: torch.Tensor, dtype: torch.dtype = torch.int8,
                 symmetric: bool = True, mode: str = "min_max",
                 dim: Union[int, None] = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Quantize the vector to int format.

    :param x: Input data tensor
    :param dtype: Quantized Vector's dtype
    :param symmetric: Whether the quantized vector space will be symmetric on 0
    :param mode: Method to find the min and max value of the data
    :param dim: Dimension to define per tensor (dim=None) or per channel (dim=dim_index)
    :return: Quantized vector, along with scale and zero point
    """
    scale, zero_point = get_scale_and_zero_point(x, symmetric=symmetric, dtype=dtype, mode=mode, dim=dim)
    return quantize_with_scale_and_zero_point(x, scale, zero_point, dtype), scale, zero_point


if __name__ == "__main__":
    at = torch.randn(3, 3)
    my_mode = "min_max"
    ct = dequantize(*quantize_int(at, mode=my_mode))
    print((ct - at).square().mean())

    for i in [0, 1]:
        ct = dequantize(*quantize_int(at, dim=i, mode=my_mode))
        print((ct - at).square().mean())
