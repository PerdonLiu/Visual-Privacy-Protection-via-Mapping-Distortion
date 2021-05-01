import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import copy


__all__ = ['PGD']


class PGD(object):
    def __init__(self, model=None, step_size=0.1, iters=1000):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.iters = iters
        self.step_size = float(step_size)
        self.loss_fn = nn.MSELoss()

    def squared_l2_norm(self, x):
        flattened = x.view(x.unsqueeze(0).shape[0], -1)
        flattened_sum = (flattened ** 2).sum(1)
        return flattened_sum

    def l2_norm(self, x):
        return self.squared_l2_norm(x).sqrt()

    def __call__(self, images, sample_images):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        batch_size = sample_images.shape[0]
        x = copy.deepcopy(sample_images.cpu().numpy())
        with torch.no_grad():
            images.requires_grad = False
            try:
                ori_outputs, ori_penultimate = self.model(images.cuda(), with_latent=True, fake_relu=True)
            except:
                ori_outputs, ori_penultimate = self.model(images.cuda(), single=False)

        for i in tqdm(range(self.iters)):
            x_var = torch.from_numpy(x).cuda()
            x_var.requires_grad = True
            try:
                outputs, penultimate = self.model(x_var, with_latent=True, fake_relu=True)
            except:
                outputs, penultimate = self.model(x_var, single=False)

            self.model.zero_grad()
            loss = self.loss_fn(penultimate, ori_penultimate)
            loss.backward()

            grad = x_var.grad
            for batch_idx in range(batch_size):
                grad_idx = grad[batch_idx]
                grad_idx_norm = self.l2_norm(grad_idx)
                grad_idx /= (grad_idx_norm + 1e-8)
                x[batch_idx] -= (self.step_size * grad_idx).cpu().numpy()
                x[batch_idx] = np.clip(x[batch_idx], 0, 1) # ensure valid pixel range

        return torch.from_numpy(x).cuda()