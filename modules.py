import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """
    1D convolutional layer with "same" padding (no downsampling),
    that is also compatible with strides > 1
    """

    def __init__(self, *args, **kwargs):
        super(Conv1dSamePadding, self).__init__(*args, **kwargs)

    def forward(self, input):
        padding = (
            self.stride[0] * (input.shape[-1] - 1)
            - input.shape[-1]
            + self.kernel_size[0]
            + (self.dilation[0] - 1) * (self.kernel_size[0] - 1)
        ) // 2
        return self._conv_forward(
            F.pad(input, (padding, padding)),
            self.weight,
            self.bias,
        )


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        device=None,
        dtype=None,
    ):
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Sequential(
            Conv1dSamePadding(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            Conv1dSamePadding(
                in_channels, out_channels, kernel_size=1, device=device, dtype=dtype
            ),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock1d(nn.Module):
    """
    Standard convolution, normalization, activation block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        activation="relu",
        dropout=0,
        depthwise=False,
    ):
        super(ConvBlock1d, self).__init__()
        assert activation is None or activation in (
            "relu",
            "tanh",
        ), "Incompatible activation function"

        conv_module = DepthwiseConv1d if depthwise else Conv1dSamePadding
        modules = [
            conv_module(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
        ]
        if activation is not None:
            modules += [nn.ReLU() if activation == "relu" else nn.Tanh()]
        if dropout > 0:
            modules += [nn.Dropout(p=dropout)]

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.squeeze(x).squeeze(-1)
        e = self.excitation(s).unsqueeze(-1)
        return x * e.expand_as(x)
