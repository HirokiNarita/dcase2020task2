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
        self.fc1 = nn.Linear(640,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3_mean = nn.Linear(256, 64) # mu
        self.fc3_var = nn.Linear(256, 64) # var
        
        self.fc4 = nn.Linear(64, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 640)

    def encoder(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)

        mean = self.fc3_mean(x)
        log_var = self.fc3_var(x)
        return mean, log_var

    def sample_z(self, mean, log_var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5*log_var)

    def decoder(self, z):
        y = F.relu(self.fc4(z))
        y = self.bn3(y)
        y = F.relu(self.fc5(y))
        y = self.bn4(y)
        y = torch.sigmoid(self.fc6(y))  
        return y
    
    def forward(self, x, device):
        mean, log_var = self.encoder(x)
        #delta = 1e-8
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        z = self.sample_z(mean, log_var, device)
        y = self.decoder(z)
        #reconstruction = torch.mean(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta))
        reconstruction = F.mse_loss(y, x)
        lower_bound = [KL, reconstruction]
        return -sum(lower_bound), z, y
