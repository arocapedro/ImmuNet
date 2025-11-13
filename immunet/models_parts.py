import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ImmuNetMainBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        strides=[1, 1],
        dilations=[1, 1],
        final_batch_norm=True,
        block_index=1,
    ):
        super().__init__()
        modules = [
            (
                f"conv{block_index}",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_sizes[0],
                    stride=strides[0],
                    dilation=dilations[0],
                    padding="valid",
                    # bias=False # bias not needed if followed by batchnorm
                ),
            ),
            (
                f"batch{block_index}",
                nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3, affine=True),
            ),  # match tensorflow values
            (f"relu{block_index}", nn.ReLU()),
            (
                f"conv{block_index+1}",
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_sizes[1],
                    stride=strides[1],
                    dilation=dilations[1],
                    padding="valid",
                    # bias=False if final_batch_norm else True # bias not needed if followed by batchnorm
                ),
            ),
        ]

        if final_batch_norm:
            modules.append(
                (
                    f"batch{block_index+1}",
                    nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3, affine=True),
                )
            )
        else:
            modules.append((f"relu{block_index+1}", nn.ReLU()))

        self.main = nn.Sequential(collections.OrderedDict(modules))

    def forward(self, x):
        return self.main(x)


class ImmuNetSkipBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        block_index=1,
        stride=1,
    ):
        super().__init__()
        self.skip = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"addconv{block_index}",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            stride=stride,
                            padding="valid",
                        ),
                    )
                ]
            )
        )

    def forward(self, x):
        return self.skip(x)


class ImmuNetEndBlock(nn.Module):
    def __init__(self, kernel_size, stride=1, dilation=1, padding=0, block_index=1):
        super().__init__()
        self.end = nn.Sequential(
            collections.OrderedDict(
                [
                    (f"relu{block_index}", nn.ReLU()),
                    (
                        f"max{block_index}",
                        nn.MaxPool2d(
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation,
                            padding=padding,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        return self.end(x)


class ImmuNetOutputBranch(nn.Module):
    def __init__(
        self, n_features, n_outputs, dropout, layer_index, kernel_size=1, stride=1
    ) -> None:
        super().__init__()
        branch_list = []
        if dropout:
            branch_list.append(("dropout", nn.Dropout(0.2)))

        branch_list.append(
            (
                f"conv{layer_index}",
                (
                    nn.Conv2d(
                        n_features, n_features, kernel_size=kernel_size, stride=stride
                    )
                ),
            )
        )
        branch_list.append((f"relu{layer_index}", nn.ReLU()))

        if dropout:
            branch_list.append(("dropout", nn.Dropout(0.2)))
        branch_list.append(
            (
                f"conv{layer_index+1}",
                nn.Conv2d(
                    n_features, n_outputs, kernel_size=kernel_size, stride=stride
                ),
            )
        )

        self.branch = nn.Sequential(collections.OrderedDict(branch_list))

    def forward(self, x):
        return self.branch(x)


class ConvSigmoid(nn.Module):
    def __init__(
        self, n_features, n_outputs, dropout, layer_index, kernel_size=1, stride=1
    ) -> None:
        super().__init__()
        branch_list = []
        if dropout:
            branch_list.append(("dropout", nn.Dropout(0.2)))

        branch_list.append(
            (
                f"conv{layer_index}",
                (
                    nn.Conv2d(
                        n_features, n_outputs, kernel_size=kernel_size, stride=stride
                    )
                ),
            )
        )
        branch_list.append((f"sigmoid{layer_index}", nn.Sigmoid()))

        self.branch = nn.Sequential(collections.OrderedDict(branch_list))

    def forward(self, x):
        return self.branch(x)
