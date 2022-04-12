import torch
import numpy as np
import cv2
import glob
import torchvision

gan_locs = ['./gan/data_gan/',
            './gan/data_ls_gan/',
            './gan/data_wgan_gp/']

# fixed_dim1, fixed_dim2 = torch.meshgrid(torch.linspace(-1, 1, 10),
#                                             torch.linspace(-1, 1, 10))

# dim1 will be along rows, dim2 along cols.

for pre in gan_locs:

    images = glob.glob(pre + 'interpolations_*.png')

    for path in images:

        print(f'Correcting - {path}')

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_steps = list(range(0, img.shape[0], img.shape[0] // 13))[: -1]
        w_steps = list(range(0, img.shape[1], img.shape[1] // 8))[: -1]

        imgs = []

        for h in h_steps:

            for w in w_steps:

                snip = img[h + 1: h + img.shape[0] // 13 - 1,
                           w + 1: w + img.shape[1] // 8 - 1]
                imgs.append(np.transpose(snip, (2, 0, 1)))

        assert len(imgs) == 104

        # print(imgs[0].shape)

        imgs = imgs[: -4]

        # Expected - 100, 32, 32, 3

        imgs = torch.Tensor(imgs).float() / 255.

        assert imgs.size(1) == 3

        torchvision.utils.save_image(imgs, path, nrow=10)




