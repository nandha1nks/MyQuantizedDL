import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantizy_utils import quantize_int, dequantize


class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super(W8A16LinearLayer, self).__init__()

        self.dtype = dtype
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (out_features, in_features), dtype=dtype)
        )
        self.register_buffer(
            "scales",
            torch.randn((out_features, 1), dtype=dtype)
        )
        self.register_buffer(
            "zero_point",
            torch.zeros((out_features, 1), dtype=dtype)
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.randn((1, out_features), dtype=dtype)
            )
        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        self.int8_weights, self.scales, self.zero_point = quantize_int(
            w_fp32, torch.int8, True, "min_max", 0
        )

        print((w_fp32 - dequantize(self.int8_weights, self.scales, self.scales)).square().mean())

    def forward(self, x):
        weights = dequantize(self.int8_weights, self.scales, self.zero_point)
        weights = weights.to(self.dtype)
        output = F.linear(x, weights)

        if self.bias is not None:
            output += self.bias
        return output


if __name__ == "__main__":
    linear = W8A16LinearLayer(3, 5)
    x = torch.randn(5, 3)
    y = linear(x)
    print(y)
