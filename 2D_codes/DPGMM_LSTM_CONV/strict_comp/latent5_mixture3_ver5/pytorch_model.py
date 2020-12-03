""" AutoEncoder in Pytorch. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import matplotlib.pyplot as plt
from pytorch_utils import do_mixup, interpolate, pad_framewise_output, to_var
import numpy as np
import sys

#import gmm 

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_normal_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(2, 2),
                              padding=(1, 1),
                              bias=False)
        #self.prelu1 = nn.PReLU()                     
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.PReLU()
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = self.act1(self.bn1(self.conv1(x)))
        #x = self.bn1(torch.tanh_(self.conv1(x)))
        return x

class FC_block(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(FC_block, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.act1 = nn.PReLU()
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn1)

        
    def forward(self, input, out=False):
        x = input
        if out == True:
            x = self.fc1(x)
        else:
            #x = self.bn1(torch.tanh_(self.fc1(x)))
            x = self.act1(self.bn1(self.fc1(x)))
        return x

class deConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(deConvBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels, 
                                        out_channels=out_channels,
                                        kernel_size=(2, 2),
                                        stride=(2, 2),
                                        #padding=(2, 2),
                                        bias=False
                                        )
        self.act1 = nn.PReLU()
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.deconv1)
        init_bn(self.bn1)

        
    def forward(self, input, out=False):
        x = input
        if out == True:
            x = self.deconv1(x)
        else:
            x = self.act1(self.bn1(self.deconv1(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=128, comp_dim=5):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim
        self.comp_dim = comp_dim
        #self.lstm_out_features = 2*hidden_dim    #(forward+backward)
        #self.input_shape = int(self.n_features * self.seq_len)

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.bn1 = nn.BatchNorm2d(self.seq_len)
        self.conv1 = ConvBlock(in_channels=2, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = FC_block(in_features=2048, out_features=512)
        self.fc2 = FC_block(in_features=512, out_features=128)
        self.fc3 = FC_block(in_features=128, out_features=32)
        self.fc4 = FC_block(in_features=32, out_features=self.comp_dim)
        #self.conv7 = ConvBlock(in_channels=2048, out_channels=4096)
        

    def forward(self, x):
        batch_size = x.shape[0]

        x, _ = self.lstm(x)
        x = x.view(batch_size, self.seq_len, 2, self.hidden_dim) # (batch_size, seq_len, channel, hidden_dim)
        #plt.imshow(x[0,:,0,:].to('cpu').detach().numpy(), aspect='auto')
        #plt.show()
        #plt.imshow(x[0,:,1,:].to('cpu').detach().numpy(), aspect='auto')
        #plt.show()
        #x = self.bn1(x) # norm -> seq direction
        x = x.transpose(1, 2)           # (batch_size, channel, seq_len, hidden_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)   # (batch, ch, 1, 1)
        x = x.view(x.shape[0], x.shape[1])   # (batch, ch)
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x, out=True)
        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=128, comp_dim=5):
        super(Decoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim
        self.comp_dim = comp_dim
        #self.lstm_out_features = 2*hidden_dim    #(forward+backward)
        #self.input_shape = int(self.n_features * self.seq_len)
        self.fc1 = FC_block(in_features=self.comp_dim, out_features=32)
        self.fc2 = FC_block(in_features=32,out_features=128)
        self.fc3 = FC_block(in_features=128,out_features=512)
        self.fc4 = FC_block(in_features=512,out_features=2048)

        self.deconv1 = deConvBlock(in_channels=2048, out_channels=1024)
        self.deconv2 = deConvBlock(in_channels=1024, out_channels=512)
        self.deconv3 = deConvBlock(in_channels=512, out_channels=256)
        self.deconv4 = deConvBlock(in_channels=256, out_channels=128)
        self.deconv5 = deConvBlock(in_channels=128, out_channels=64)
        self.deconv6 = deConvBlock(in_channels=64, out_channels=1)

        #self.lstm = nn.LSTM(
        #    input_size=self.n_features*2,
        #    hidden_size=self.hidden_dim*2,
        #    num_layers=3,
        #    batch_first=True,
        #    bidirectional=False,
        #    dropout=0.2
        #)

        #self.fc1 = nn.Linear(2*128, 128)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)    # (batch, ch, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x, out=True)
        #x = self.deconv7(x) # (batch_size, seq_len, channel, hidden_dim)
        x = x.view(batch_size, self.seq_len, self.n_features)
        #x = torch.sigmoid_(x)
        #print(x)
        #x, _ = self.lstm(x)
        #print(x.shape)
        #x = x.view(batch_size, self.seq_len, self.n_features*2) # (batch_size, seq_len, hidden_dim)
        #x = torch.sigmoid_(self.fc1(x))
        #x = x.view(batch_size, self.seq_len, self.n_features)
        return x

class Estimation_net(nn.Module):
    def __init__(self, input_dim, mid_dim, mixtures):
        super(Estimation_net, self).__init__()
        self.input_dim, self.mid_dim, self.mixtures = input_dim, mid_dim, mixtures
        
        #self.bn0 = nn.BatchNorm1d(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, self.mid_dim)
        self.bn1 = nn.BatchNorm1d(self.mid_dim)
        self.fc2 = nn.Linear(self.mid_dim, self.mixtures)
        self.bn2 = nn.BatchNorm1d(self.mixtures)
        self.act1 = nn.PReLU()
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        #init_bn(self.bn0)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x):
        #x = self.bn0(x)
        x = self.act1(self.bn1(self.fc1(x)))
        x = F.softmax(self.bn2(self.fc2(x)), dim=1)
        return x

eps = torch.autograd.Variable(torch.cuda.FloatTensor([1.e-8]), requires_grad=False)
def relative_euclidean_distance(x1, x2, eps=eps):
    """x1 and x2 are assumed to be Variables or Tensors.
    They have shape [batch_size, dimension_embedding]"""
    num = torch.norm(x1 - x2, p=2, dim=1)  # dim [batch_size]
    denom = torch.norm(x1, p=2, dim=1)  # dim [batch_size]
    return num / torch.max(denom, eps)

def cosine_similarity(x1, x2, eps=eps):
    """x1 and x2 are assumed to be Variables or Tensors.
    They have shape [batch_size, dimension_embedding]"""
    dot_prod = torch.sum(x1 * x2, dim=1)  # dim [batch_size]
    dist_x1 = torch.norm(x1, p=2, dim=1)  # dim [batch_size]
    dist_x2 = torch.norm(x2, p=2, dim=1)  # dim [batch_size]
    return dot_prod / torch.max(dist_x1*dist_x2, eps)

class DAGMM(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, latent_size, mixture_size):
        
        super(DAGMM, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #    freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm1d(64)
        
        self.encoder = Encoder(seq_len=64, n_features=64, hidden_dim=64, comp_dim=latent_size)

        self.decoder = Decoder(seq_len=64, n_features=64, hidden_dim=64, comp_dim=latent_size)
        self.bn1 = nn.BatchNorm1d(latent_size+1)
        #self.meta_dense = Meta_dense(input_dim=1024, mid_dim=512, comp_dim=latent_size)
        #self.reconst_bn1 = nn.BatchNorm1d(1)
        #self.reconst_bn2 = nn.BatchNorm1d(1)
        self.estimation = Estimation_net(input_dim=latent_size+1, mid_dim=int((latent_size+1)*2), mixtures=mixture_size)
        self.bn1 = nn.BatchNorm1d(latent_size+1)
        #self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def calc_reconstruction(self, x, x_dash):
        squared_euclidean = torch.sum(torch.square(x - x_dash), dim=(1,2))+1e-12
        return squared_euclidean.unsqueeze(-1) # unsqueezeを要確認

    def preproc_latent(self, z_c):
        # z_c dim : (N, C, H, W) = 4
        z_c = torch.sum(z_c, dim=(2,3)) # (N, C, H+W) = 3
        ch_dim = z_c.size()[1]
        z_c = torch.reshape(z_c, (-1, ch_dim)) # (N*(H+W), ch) = 2
        return z_c

    def forward(self, input, device='cuda:0'):
        # preproc
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = torch.squeeze(x, 1)         # (batch_size, time_steps, mel_bins)
        x = x[:, :64, :]   # trim
        input_img = x        # save input melspectrogram
        # network
        x = self.bn0(x)
        enc = self.encoder(x)
        dec = self.decoder(enc)

        # calc reconstruction
        rec_euclidean = self.calc_reconstruction(input_img, dec)

        # calc latent
        #enc = self.preproc_latent(enc)
        #z_c = self.meta_dense(enc)
        #meta_euclidean = self.relative_euclidean_distance(enc, meta_hat).unsqueeze(-1)
        #meta_loss = F.mse_loss(meta_hat, enc)
        z = torch.cat([enc, rec_euclidean], dim=1) # unuse cosine_similarity
        z = torch.log(1e3+z)
        #z = F.normalize(z, p=2, dim=0)
        #z = torch.log(1e3+z)
        z = self.bn1(z)
        plt.hist(z[:,0].data.cpu().numpy())
        plt.show()
        #print(z)
        # estimation net
        gamma = self.estimation(z)
        return {'x':input_img, 'x_hat':dec, 'z_c':enc, 'z':z, 'gamma':gamma, 'enc':enc}

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy()* (2*np.pi))
            determinant = np.prod(np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None))
            det_cov.append(determinant)
            #det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / (cov_k.diag()))

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        #det_cov = torch.cat(det_cov).cuda()
        det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))
        #print(cov_k.shape)
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val + eps)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov) + eps).unsqueeze(0), dim = 1) + eps)
        #sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)
        if size_average:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = F.mse_loss(x_hat, x)
        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag
