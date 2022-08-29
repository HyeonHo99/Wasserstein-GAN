import torch
import torch.nn

def gradient_penalty(critic,real,fake,device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = alpha * real + (1-alpha) * fake

    ## Calculate critic scores
    interpolated_images_critic = critic(interpolated_images)

    ## Calculate gradients
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_images_critic,
        grad_outputs=torch.ones_like((interpolated_images_critic)),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.reshape(BATCH_SIZE,-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty