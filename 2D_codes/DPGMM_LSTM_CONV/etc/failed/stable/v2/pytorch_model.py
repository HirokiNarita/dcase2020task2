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

#import gmm 

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
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
        x = self.bn1(self.act1(self.conv1(x)))
        #if pool_type == 'max':
        #    x = F.max_pool2d(x, kernel_size=pool_size)
        #elif pool_type == 'avg':
        #    x = F.avg_pool2d(x, kernel_size=pool_size)
       # elif pool_type == 'avg+max':
        #    x1 = F.avg_pool2d(x, kernel_size=pool_size)
        #    x2 = F.max_pool2d(x, kernel_size=pool_size)
        #    x = x1 + x2
        #else:
        #    raise Exception('Incorrect argument!')
        
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

        
    def forward(self, input):
        x = input
        x = self.bn1(self.act1(self.deconv1(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=128):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim
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
        #self.conv6 = ConvBlock(in_channels=1024, out_channels=2048)
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
        #print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)
        #x = self.conv7(x)
        #print(x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=128):
        super(Decoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim
        #self.lstm_out_features = 2*hidden_dim    #(forward+backward)
        #self.input_shape = int(self.n_features * self.seq_len)
        #self.deconv1 = deConvBlock(in_channels=2048, out_channels=1024)
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
        #x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
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

class metaDense(nn.Module):
    def __init__(self, input_dim, mid_dim, comp_dim):

        super(metaDense, self).__init__()
        self.input_dim, self.mid_dim, self.comp_dim = input_dim, mid_dim, comp_dim

        self.bn0 = nn.BatchNorm1d(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, self.mid_dim)
        #self.mid_dim = int(self.mid_dim / 2)
        self.bn1 = nn.BatchNorm1d(self.mid_dim)
        self.fc2 = nn.Linear(self.mid_dim, int(self.mid_dim / 2))
        self.mid_dim = int(self.mid_dim / 2)
        self.bn2 = nn.BatchNorm1d(self.mid_dim)
        self.fc3 = nn.Linear(self.mid_dim, comp_dim)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        #self.act3 = nn.PReLU()

        #self.init_weight()
        
    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_bn(self.bn0)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, x):
        x = self.bn0(x)
        x = self.act1(self.fc1(x))
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.bn1(x)
        x = self.act2(self.fc2(x))
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.bn2(x)
        x = torch.tanh_(self.fc3(x))

        return x

class Estimation_net(nn.Module):
    def __init__(self, input_dim, mid_dim, mixtures):
        super(Estimation_net, self).__init__()
        self.input_dim, self.mid_dim, self.mixtures = input_dim, mid_dim, mixtures
        
        #self.bn0 = nn.BatchNorm1d(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, self.mid_dim)
        #self.bn1 = nn.BatchNorm1d(self.mid_dim)
        self.fc2 = nn.Linear(self.mid_dim, self.mixtures)
        self.act1 = nn.PReLU()
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        #init_bn(self.bn0)
        #init_bn(self.bn1)



    def forward(self, x):
        #x = self.bn0(x)
        x = self.act1(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Cholesky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_tensors
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

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
        
        self.encoder = Encoder(seq_len=64, n_features=64, hidden_dim=64)

        self.decoder = Decoder(seq_len=64, n_features=64, hidden_dim=64)

        self.metaDense = metaDense(input_dim=1024, mid_dim=512, comp_dim=latent_size)
        self.reconst_bn1 = nn.BatchNorm1d(1)
        self.reconst_bn2 = nn.BatchNorm1d(1)
        self.estimation = Estimation_net(input_dim=latent_size+1, mid_dim=int((latent_size+1)/2), mixtures=mixture_size)
        self.bn1 = nn.BatchNorm1d(latent_size+1)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
    
    def calc_reconstruction(self, x, x_dash):
        def relative_euclidean_distance(a, b):
            return (a-b).norm(2, dim=1) / a.norm(2, dim=1)
        
        x = torch.sum(x, dim=(1,2)).unsqueeze(-1)
        x = self.reconst_bn1(x)
        #x = F.normalize(x, dim=0, p=2)
        x_dash = torch.sum(x_dash, dim=(1,2)).unsqueeze(-1)
        x_dash = self.reconst_bn2(x)
        #x_dash = F.normalize(x_dash, dim=0, p=2)
        #print(x)
        #print(x_dash)
        rec_cosine = F.cosine_similarity(x, x_dash, dim=1)
        rec_euclidean = relative_euclidean_distance(x, x_dash)
        #print(rec_cosine)
        #print(rec_euclidean)
        return rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1) # unsqueezeを要確認

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
        # scaling
        x = F.normalize(x, dim=(1,2), p=2)
        #plt.imshow(x[0,:,:].data.cpu().numpy(), aspect="auto")
        #plt.show()
        #x_min, x_max = torch.min(x, dim=2, keepdim=True)[0], torch.max(x, dim=2, keepdim=True)[0]
        #x = (x - x_min) / (x_max - x_min)
        #del x_min, x_max
        input_img = x        # save input melspectrogram
        # network
        x = self.bn0(x)
        enc = self.encoder(x)
        dec = torch.tanh_(self.decoder(enc)) # torch.sigmoid_(
        #print(dec.shape)
        # calc reconstruction
        rec_euclidean, rec_cosine = self.calc_reconstruction(input_img, dec)
        #print(rec_euclidean.mean())
        # calc latent
        #print(enc.shape)
        enc = self.preproc_latent(enc)
        z_c = self.metaDense(enc)
        #print(z_c[0])
        #print(z_c.shape)
        #print(rec_euclidean.shape)
        #print(rec_cosine.shape)
        z = torch.cat([z_c, rec_euclidean], dim=1) # unuse cosine_similarity
        z = self.bn1(z)
        #print(z)
        # estimation net
        gamma = self.estimation(z)
        #print(gamma[0])
        #return enc, dec, z, gamma
        return {'x':input_img, 'x_hat':dec, 'z_c':z_c, 'z':z, 'gamma':gamma}
    
    # def add_noise(mat, stdev=0.001):
    #     """
    #     :param mat: should be of shape(k, d, d)
    #     :param stdev: the standard deviation of noise
    #     :return: a matrix with little noises
    #     """
    #     with tf.name_scope('gaussian_noise'):
    #         dims = mat.get_shape().as_list()[1]
    #         noise = stdev + tf.random_normal([dims], 0, stdev * 1e-1)
    #         noise = tf.diag(noise)
    #         noise = tf.expand_dims(noise, axis=0)
    #         noise = tf.tile(noise, (mat.get_shape()[0], 1, 1))
    #     return mat + noise

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
        eps = 1e-6
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / (cov_k.diag()))

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov).cuda()
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))
        #print(cov_k.shape)
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val + eps)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        #sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)
        sample_energy = sample_energy
        sample_energy_var = sample_energy.var()
        if size_average:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag, sample_energy_var

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = F.mse_loss(x_hat, x)
        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag, sample_energy_var = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag
