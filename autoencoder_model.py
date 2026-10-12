"""Convolutional autoencoder (method A)."""

import torch
from torch import nn

import configs


class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_channels: int = configs.BOTTLENECK_CHANNELS) -> None:
        super().__init__()
        channels = (*configs.ENCODER_CHANNELS, bottleneck_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels[0], configs.CONV_KERNEL, configs.CONV_STRIDE, configs.CONV_PADDING),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[1], configs.CONV_KERNEL, configs.CONV_STRIDE, configs.CONV_PADDING),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], configs.CONV_KERNEL, configs.CONV_STRIDE, configs.CONV_PADDING),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], configs.CONV_KERNEL, configs.CONV_STRIDE, configs.CONV_PADDING),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[4], configs.CONV_KERNEL, configs.CONV_STRIDE, configs.CONV_PADDING),
            nn.ReLU(inplace=True),
        )
        decoder_channels = list(reversed(channels))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_channels[0],
                decoder_channels[1],
                configs.CONV_KERNEL,
                configs.CONV_STRIDE,
                configs.CONV_PADDING,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                decoder_channels[1],
                decoder_channels[2],
                configs.CONV_KERNEL,
                configs.CONV_STRIDE,
                configs.CONV_PADDING,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                decoder_channels[2],
                decoder_channels[3],
                configs.CONV_KERNEL,
                configs.CONV_STRIDE,
                configs.CONV_PADDING,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                decoder_channels[3],
                decoder_channels[4],
                configs.CONV_KERNEL,
                configs.CONV_STRIDE,
                configs.CONV_PADDING,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                decoder_channels[4],
                3,
                configs.CONV_KERNEL,
                configs.CONV_STRIDE,
                configs.CONV_PADDING,
            ),
            nn.Sigmoid(),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(batch))
