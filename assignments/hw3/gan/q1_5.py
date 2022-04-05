import os
import torch

from networks import Discriminator, Generator
from train import train_model

def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    # loss = max_D E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]

    grad = torch.autograd.grad(discrim_interp,
                               interp,
                               grad_outputs=torch.ones_like(
                                   discrim_interp).cuda(),
                               create_graph=True, 
                               retain_graph=True)[0]
    grad = grad.reshape(interp.size(0), -1)

    return torch.mean(discrim_fake) - torch.mean(discrim_real) + \
        lamb * torch.mean((torch.linalg.norm(grad, dim=-1) - 1) ** 2)


def compute_generator_loss(discrim_fake):
    # TODO 1.5.1: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    return - torch.mean(discrim_fake)


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
    )
