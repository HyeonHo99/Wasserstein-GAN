import torch
import torch.nn as nn
from critic import Critic
from generator import Generator
from utils import initialize_weights

def test():
    N, img_channel, H, W = 8, 3, 64, 64
    z_dim = 100

    x = torch.randn((N,img_channel,H,W))
    z = torch.randn((N,z_dim,1,1))

    critic = Critic(img_channel,8)
    gen = Generator(z_dim,img_channel,8)
    initialize_weights(critic)
    initialize_weights(gen)

    assert critic(x).shape == (N,1,1,1), "Testing Discriminator failed"
    assert gen(z).shape == (N,3,64,64), "Testing Generator failed"

    print("success")

test()