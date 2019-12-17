import torch
from .utiil import MMDStatistic

from trvaep.utils import partition


def MSE_kl(recon_x, x, mu, logvar, alpha=.1):
    mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), )
    loss = (mse_loss + alpha * kl_loss) / x.size(0)
    return loss, mse_loss / x.size(0), (alpha * kl_loss) / x.size(0)


def mmd(n_conditions, beta):
    def mmd_loss(x, y):
        alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
        conditions_mmd = partition(x, y, n_conditions)
        loss = 0.0
        for i in range(len(conditions_mmd)):
            for j in range(i):
                mmd_calculator = MMDStatistic(conditions_mmd[j].size(0), conditions_mmd[j + 1].size(0))
                loss += mmd_calculator(conditions_mmd[j], conditions_mmd[j + 1], alphas=alphas)
        return beta * loss

    return mmd_loss
