import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from critic import Critic
from generator import Generator
from utils import gradient_penalty, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Hyperparameters
LR = 1e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 30
CHANNELS_DISC = 64
CHANNELS_GEN = 64
N_CRITICS = 5
WEIGHT_CLIP = 0.01
LAMBDA_GP = 10

## Data
transforms = transforms.Compose(
    [ transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)]) ]
)

dataset = datasets.ImageFolder(root="celebA/female",transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

## models
critic = Critic(img_channel=CHANNELS_IMG, features_d=CHANNELS_DISC).to(device)
gen = Generator(z_dim=Z_DIM, img_channel=CHANNELS_IMG, features_g=CHANNELS_GEN).to(device)
initialize_weights(critic)
initialize_weights(gen)

## optimizer, loss

opt_critic = optim.Adam(critic.parameters(), lr=LR,betas=(0.0,0.9))
opt_gen = optim.Adam(gen.parameters(), lr=LR,betas=(0.0,0.9))

## for tensorboard
fixed_latent = torch.randn(32,Z_DIM,1,1).to(device)
writer_fake = SummaryWriter(f"runs/DCGAN_celebA/fake")
writer_real = SummaryWriter(f"runs/DCGAN_celebA/real")
writer_loss = SummaryWriter(f"runs/DCGAN_celebA/loss")
step = 0
loss_step = 0

critic.train()
gen.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx,(real,_) in enumerate(loader):
        real = real.to(device)
        batch_size = real.shape[0]

        ## TRAIN CRITIC
        for _ in range(N_CRITICS):
            latent_vector = torch.randn(batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(latent_vector)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            ## WGAN CRITIC LOSS + Gradient Penalty
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = -1*(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            ## GRADIENT PENALTY [-0.01, 0.01]
            for weight in critic.parameters():
                weight.data.clamp_(-WEIGHT_CLIP,WEIGHT_CLIP)

        ## TRAIN GENERATOR
        output = critic(fake)
        loss_gen = -1*torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        ## write loss on tensorboard on every 'step'
        writer_loss.add_scalar("Discriminator Loss", loss_critic, global_step=loss_step)
        writer_loss.add_scalar("Generator Loss", loss_gen, global_step=loss_step)
        loss_step += 1

        ## show images on tensorboard on every 'epoch'
        if batch_idx % 100 == 0:
            print(f"EPOCH {epoch}/{NUM_EPOCHS} BATCH {batch_idx}/{len(loader)} LOSS D:{loss_critic:.4f} LOSS G:{loss_gen:.4f}")

            with torch.no_grad():
                generated = gen(fixed_latent).reshape(-1,CHANNELS_IMG,IMAGE_SIZE,IMAGE_SIZE)
                real = real[:32]

                img_grid_fake = torchvision.utils.make_grid(generated,normalize=True)
                img_grid_real = torchvision.utils.make_grid(real,normalize=True)

                writer_fake.add_image("Fake",img_grid_fake,global_step=step)
                writer_real.add_image("Real",img_grid_real,global_step=step)
            step += 1
