import torch
import torchvision
from cleanfid import fid
from matplotlib import pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # Concretely, for the first two dimensions of the latent space
    # generate a grid of points that range from -1 to 1 on each dimension (10 points for each dimension).
    # hold the rest of z to be some fixed random value. Forward the generated samples through the generator
    # and save out an image holding all 100 samples.
    # use torchvision.utils.save_image to save out the visualization.

    # z = torch.rand((1, 128)).repeat(100, 1)
    z = torch.normal(0, 1., (1, 128)).repeat(100, 1)

    fixed_dim1, fixed_dim2 = torch.meshgrid(torch.linspace(-1, 1, 10),
                                            torch.linspace(-1, 1, 10))
    fixed_dims = torch.hstack((fixed_dim1.reshape(-1, 1),
                               fixed_dim2.reshape(-1, 1)))
    z[..., :2] = fixed_dims

    # Expected out - 100, 3, 32, 32
    out = gen.forward_given_samples(z.cuda())

    torchvision.utils.save_image(out, path)


