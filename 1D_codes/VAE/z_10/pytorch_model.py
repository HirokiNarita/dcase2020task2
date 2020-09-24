""" AutoEncoder in Pytorch. """
import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    """
        AutoEncoder
    """

    def __init__(self):
        super(AutoEncoder, self).__init__()

        layers = [nn.Linear(640, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 8),
                  nn.Linear(8, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 128),
                  nn.Linear(128, 640), ]
        self.layers = nn.ModuleList(layers)

        bnorms = [nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(8),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128),
                  nn.BatchNorm1d(128), ]
        self.bnorms = nn.ModuleList(bnorms)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        hidden = self.relu(self.bnorms[0](self.layers[0](inputs)))  # 640->128
        hidden = self.relu(self.bnorms[1](self.layers[1](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[2](self.layers[2](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[3](self.layers[3](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[4](self.layers[4](hidden)))  # 128->8
        hidden = self.relu(self.bnorms[5](self.layers[5](hidden)))  # 8->128
        hidden = self.relu(self.bnorms[6](self.layers[6](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[7](self.layers[7](hidden)))  # 128->128
        hidden = self.relu(self.bnorms[8](self.layers[8](hidden)))  # 128->128
        output = self.layers[9](hidden)                             # 128->640

        return output

class VAE(nn.Module):
    """
        Variational AutoEncoder
    """

    def __init__(self):
        super().__init__()
        
        # vae
        self.efc1 = nn.Linear(640, 128)
        self.efc2 = nn.Linear(128, 128)
        self.efc3 = nn.Linear(128, 128)
        self.efc4 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128,10) # mu
        self.fc_var = nn.Linear(128, 10) # var
        self.dfc1 = nn.Linear(10, 128)
        self.dfc2 = nn.Linear(128, 128)
        self.dfc3 = nn.Linear(128, 128)
        self.dfc4 = nn.Linear(128, 128)
        self.dfc5 = nn.Linear(128, 640)

        b_norms = [
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(128)
        ]
        self.b_norms = nn.ModuleList(b_norms)
        self.relu = nn.ReLU()
    def encoder(self, x):
        x = self.relu(self.b_norms[0](self.efc1(x)))
        x = self.relu(self.b_norms[1](self.efc2(x)))
        x = self.relu(self.b_norms[2](self.efc3(x)))
        x = self.relu(self.b_norms[3](self.efc4(x)))

        mean = self.fc_mean(x)
        var = self.fc_var(x)
        
        return mean, var

    def sample_z(self, mean, var, device):
        delta = 1e-8
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5*var+delta)

    def decoder(self, z):
        y = self.relu(self.dfc1(z))
        y = self.relu(self.b_norms[4](self.dfc2(y)))
        y = self.relu(self.b_norms[5](self.dfc3(y)))
        y = self.relu(self.b_norms[6](self.dfc4(y)))
        y = self.dfc5(y)
        return y
    
    def forward(self, x, device):
        mean, var = self.encoder(x)
        delta = 1e-8
        KL = 0.5 * torch.sum(1 + var - mean**2 - torch.exp(var+delta))
        z = self.sample_z(mean, var, device)
        y = self.decoder(z)
        #reconstruction = torch.mean(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta))
        reconstruction = F.mse_loss(y, x)
        lower_bound = [KL, reconstruction]
        loss = reconstruction - KL
        return loss, lower_bound, z, y
