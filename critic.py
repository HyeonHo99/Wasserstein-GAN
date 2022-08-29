import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, img_channel, features_d):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
                                                                                        ## initial input shape: N x 3 x 64 x 64
            nn.Conv2d(img_channel, features_d, kernel_size=4, stride=2, padding=1),      ## channel_d x 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),                               ## channel_d*2 x 16 x 16
            self._block(features_d*2, features_d*4, 4, 2, 1),                             ## channel_d*4 x 8 x 8
            self._block(features_d*4, features_d*8, 4, 2, 1),                             ## channel_d*8 x 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0)              ## 1 x 1 x 1
            # nn.Sigmoid()
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,bias=False),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)