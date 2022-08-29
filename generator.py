import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_channel, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            ## initial input shape: N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0),          ## channel_g*16 x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1),    ## channel_g*8 x 8 x 8
            self._block(features_g*8, features_g*4, 4, 2, 1),     ## channel_g*4 x 16 x 16
            self._block(features_g*4, features_g*2, 4, 2, 1),     ## channel_g*2 x 32 x 32
            nn.ConvTranspose2d(in_channels=features_g*2, out_channels=img_channel, kernel_size=4, stride=2, padding=1),   ## 3 x 64 x 64
            nn.Tanh()
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.gen(x)