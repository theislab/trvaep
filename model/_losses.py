import torch


def MSE_kl(recon_x, x, mu, logvar, alpha=.1):
    mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),)
    loss = (mse_loss + alpha * kl_loss)/ x.size(0)
    return loss, mse_loss/x.size(0), (alpha*kl_loss)/x.size(0)
