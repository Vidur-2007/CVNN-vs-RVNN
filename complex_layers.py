"""
Minimal, self-contained complex-valued neural network layers in PyTorch.

A complex tensor is represented as a pair of real tensors (re, im) of
identical shape (the "dual real tensor" trick), following the standard
formulation used e.g. in Trabelsi et al., "Deep Complex Networks" (ICLR 2018)
and Zhang et al. for complex-valued SAR/radar networks.

Complex multiplication (a+bi)(c+di) = (ac-bd) + (ad+bc)i is implemented with
real convolutions/linear layers sharing weights across the real and
imaginary parts, which is what gives a complex layer its algebraic
structure (and its parameter count of ~2x a same-width real layer, since it
stores a real weight matrix AND an imaginary weight matrix).
"""
import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv_r = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.conv_i = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        re, im = x
        # (re + i*im) conv (Wr + i*Wi) = (re*Wr - im*Wi) + i*(re*Wi + im*Wr)
        out_re = self.conv_r(re) - self.conv_i(im)
        out_im = self.conv_r(im) + self.conv_i(re)
        return (out_re, out_im)


class ComplexLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.lin_r = nn.Linear(in_f, out_f, bias=bias)
        self.lin_i = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        re, im = x
        out_re = self.lin_r(re) - self.lin_i(im)
        out_im = self.lin_r(im) + self.lin_i(re)
        return (out_re, out_im)


class ComplexBatchNorm2d(nn.Module):
    """Simplified ('naive') complex batch norm: independently whitens the
    real and imaginary channels. This is the lightweight variant; the full
    covariance-whitening version in Trabelsi et al. normalizes the 2x2
    real/imag covariance jointly. We use the naive version for parameter-
    count parity with the real-valued baselines' BatchNorm2d."""
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x):
        re, im = x
        return (self.bn_r(re), self.bn_i(im))


class ModReLU(nn.Module):
    """modReLU (Arjovsky et al. 2016 / Trabelsi et al. 2018): applies a
    ReLU to the *magnitude* of the complex number (with a learnable bias)
    while preserving phase exactly. This is a natively phase-equivariant
    nonlinearity: rotating the input by e^{i*phi} rotates the output by the
    same e^{i*phi}. A real-valued network has no equivalent built-in
    operation."""
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1) - 0.1)

    def forward(self, x):
        re, im = x
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-8)
        scale = torch.relu(mag + self.bias) / mag
        return (re * scale, im * scale)


def complex_magnitude(x):
    re, im = x
    return torch.sqrt(re ** 2 + im ** 2 + 1e-8)


def complex_avg_pool(x):
    re, im = x
    return (re.mean(dim=(2, 3)), im.mean(dim=(2, 3)))
