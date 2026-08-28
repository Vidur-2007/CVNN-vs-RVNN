"""
Three parameter-matched CNN classifiers for the 6-class complex-spectrogram
task:

  RVNN_mag - real-valued CNN, sees ONLY the magnitude spectrogram (1 ch).
             The common "throw away the phase" baseline used in a lot of
             audio/vision-style pipelines applied naively to complex data.

  RVNN_ri  - real-valued CNN, sees real & imaginary parts as 2 input
             channels. Has full access to the same information as the CVNN,
             but no architectural phase structure (an ordinary real conv
             mixes the two channels with no constraint) -- this isolates
             the "given phase info" leg of the argument from the
             "biased/equivariant complex algebra" leg.

  CVNN     - native complex-valued CNN (complex conv / complex batchnorm /
             modReLU from complex_layers.py), final classification head
             reads out the complex-magnitude of the last feature map, which
             makes the whole network equivariant to a global phase rotation
             of the input.

Widths are chosen so all three models have closely matched parameter
counts (see param_report() at the bottom / count_params.py).
"""
import torch
import torch.nn as nn
from complex_layers import (
    ComplexConv2d, ComplexBatchNorm2d, ModReLU, complex_magnitude, complex_avg_pool,
)

NUM_CLASSES = 6


class RVNN_Mag(nn.Module):
    def __init__(self, w=(17, 34, 68)):
        super().__init__()
        w1, w2, w3 = w
        self.net = nn.Sequential(
            nn.Conv2d(1, w1, 3, 1, 1), nn.BatchNorm2d(w1), nn.ReLU(inplace=True),
            nn.Conv2d(w1, w2, 3, 2, 1), nn.BatchNorm2d(w2), nn.ReLU(inplace=True),
            nn.Conv2d(w2, w3, 3, 2, 1), nn.BatchNorm2d(w3), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(w3, NUM_CLASSES)

    def forward(self, mag):
        h = self.net(mag)
        h = h.mean(dim=(2, 3))
        return self.fc(h)


class RVNN_RI(nn.Module):
    def __init__(self, w=(17, 34, 68)):
        super().__init__()
        w1, w2, w3 = w
        self.net = nn.Sequential(
            nn.Conv2d(2, w1, 3, 1, 1), nn.BatchNorm2d(w1), nn.ReLU(inplace=True),
            nn.Conv2d(w1, w2, 3, 2, 1), nn.BatchNorm2d(w2), nn.ReLU(inplace=True),
            nn.Conv2d(w2, w3, 3, 2, 1), nn.BatchNorm2d(w3), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(w3, NUM_CLASSES)

    def forward(self, ri):  # ri: (B, 2, H, W) real+imag stacked
        h = self.net(ri)
        h = h.mean(dim=(2, 3))
        return self.fc(h)


class CVNN(nn.Module):
    def __init__(self, w=(12, 24, 48)):
        super().__init__()
        w1, w2, w3 = w
        self.conv1 = ComplexConv2d(1, w1, 3, 1, 1)
        self.bn1 = ComplexBatchNorm2d(w1)
        self.act1 = ModReLU(w1)
        self.conv2 = ComplexConv2d(w1, w2, 3, 2, 1)
        self.bn2 = ComplexBatchNorm2d(w2)
        self.act2 = ModReLU(w2)
        self.conv3 = ComplexConv2d(w2, w3, 3, 2, 1)
        self.bn3 = ComplexBatchNorm2d(w3)
        self.act3 = ModReLU(w3)
        self.fc = nn.Linear(w3, NUM_CLASSES)

    def forward(self, x):  # x: (re, im) each (B, 1, H, W)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        mag = complex_magnitude(x)          # (B, w3, H', W')
        h = mag.mean(dim=(2, 3))            # phase-rotation-invariant readout
        return self.fc(h)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    mag_net = RVNN_Mag()
    ri_net = RVNN_RI()
    cvnn = CVNN()
    print("RVNN_mag params:", count_params(mag_net))
    print("RVNN_ri  params:", count_params(ri_net))
    print("CVNN     params:", count_params(cvnn))

    B = 4
    re = torch.randn(B, 1, 32, 32)
    im = torch.randn(B, 1, 32, 32)
    mag = torch.sqrt(re**2 + im**2)
    ri = torch.cat([re, im], dim=1)
    print("mag out", mag_net(mag).shape)
    print("ri  out", ri_net(ri).shape)
    print("cvnn out", cvnn((re, im)).shape)
