import torch
import torch.nn as nn
from .helper_module import Encoder, Decoder


class CVAE(nn.Module):
    def __init__(self, input_dim, num_classes=None, encoder_layer_sizes=[128],
                 latent_dim=10, decoder_layer_sizes=[128], alpha=0.001, use_batch_norm=True,
                 dr_rate=0.2):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_dim) == int
        assert type(decoder_layer_sizes) == list
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_cls = num_classes
        self.dr_rate = dr_rate
        if self.dr_rate>0:
            self.use_dr = True
        else:
            self.use_dr = False
        self.use_bn = use_batch_norm
        self.alpha = alpha
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes.append(self.input_dim)
        self.encoder = Encoder(encoder_layer_sizes, self.latent_dim,
                                self.use_bn, self.use_dr, self.dr_rate, self.num_cls)
        self.decoder = Decoder(decoder_layer_sizes, self.latent_dim,
                               self.use_bn, self.use_dr, self.dr_rate, self.num_cls,)

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_dim])
        recon_x = self.decoder(z, c)
        return recon_x

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def get_latent(self, x, c=None, mean=False):
        if c is not None:
            c = torch.tensor(c)
        x = torch.tensor(x)
        z_mean, z_log_var = self.encoder(x, c)
        z_sample = self.sampling(z_mean, z_log_var)
        if mean:
            return z_mean.data.numpy()
        return z_sample.data.numpy()

    def reconstruct(self, x, c=None, use_latent=False):
        if use_latent:
            x = torch.tensor(x)
            if c is not None:
                c = torch.tensor(c)
            reconstructed = self.decoder(x, c)
            return reconstructed.data.numpy()
        else:
            z = self.get_latent(x, c)
            z = torch.tensor(z)
            if c is not None:
                c = torch.tensor(c)
            reconstructed = self.decoder(z, c)
            return reconstructed.data.numpy()
        
    def forward(self, x, c=None):
        z_mean, z_log_var = self.encoder(x, c)
        z = self.sampling(z_mean, z_log_var)
        recon_x = self.decoder(z, c)
        return recon_x, z_mean, z_log_var


