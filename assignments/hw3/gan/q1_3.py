import os

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.3.1: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.

    real_labels = torch.ones_like(discrim_real)
    fake_labels = torch.zeros_like(discrim_fake)

    return torch.nn.BCEWithLogitsLoss()(torch.hstack((discrim_real,
                                                      discrim_fake)),
                                        torch.hstack((real_labels,
                                                      fake_labels)))


def compute_generator_loss(discrim_fake):
    # TODO 1.3.1: Implement GAN loss for generator.
    # Maximize log D(G(z)) - forcing all labels to be 1.

    return torch.nn.BCEWithLogitsLoss()(discrim_fake,
                                        torch.ones_like(discrim_fake))


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # gen = torch.jit.load('data_gan/generator.pt', map_location='cuda')
    # disc = torch.jit.load('data_gan/discriminator.pt', map_location='cuda')

    # TODO 1.3.2: Run this line of code.
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
