""" AutoEncoder in Pytorch. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from pytorch_utils import do_mixup, interpolate, pad_framewise_output

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

class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock5x5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class deConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(deConvBlock5x5, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels, 
                                        out_channels=out_channels,
                                        kernel_size=(2, 2),
                                        stride=(2, 2),
                                        #padding=(2, 2),
                                        bias=False
                                        )
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.deconv1)
        init_bn(self.bn1)

        
    def forward(self, input, out_layer=False):
        
        x = input
        if out_layer == True:
            x = self.deconv1(x)
        else:
            x = F.relu_(self.bn1(self.deconv1(x)))
        
        return x

class CNN6PANNsVAE(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        
        super(CNN6PANNsVAE, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        #mid_layer_size = (512, 64, 8)

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

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        
        #flatten_size = mid_layer_size[0] * mid_layer_size[1] * mid_layer_size[2]
        self.fc_mean = nn.Linear(65536, 64) # mu
        self.fc_var = nn.Linear(65536, 64)  # var
        self.fc_z = nn.Linear(64, 65536)    # z
        

        self.deconv_block1 = deConvBlock5x5(in_channels=512, out_channels=256)
        self.deconv_block2 = deConvBlock5x5(in_channels=256, out_channels=128)
        self.deconv_block3 = deConvBlock5x5(in_channels=128, out_channels=64)
        self.deconv_block4 = deConvBlock5x5(in_channels=64, out_channels=1)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
    
    def encoder(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x

    def bottle_neck(self, x):
        mean = self.fc_mean(x)
        var = self.fc_var(x)
        return mean, var
    
    def sample_z(self, mean, var, device):
        delta = 1e-8
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5*var+delta)

    def decoder(self, x):
        x = self.deconv_block1(x)
        x = self.deconv_block2(x)
        x = self.deconv_block3(x)
        x = self.deconv_block4(x, out_layer=True)
        return x

    def forward(self, input, device, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3) # (batch_size, melbins, time_steps, 1)
        x = x[:, :, :512,:]   # trim
        input_spec = x        # input melspectrogram
        x = self.bn0(x)
        x = x.transpose(1, 3) # (batch_size, 1, time_steps, melbins)
        # encoder
        x = self.encoder(x)
        # flatten
        middle_size = x.size()
        ### middle ###
        flatten_size = middle_size[1] * middle_size[2] * middle_size[3]
        x = x.view(middle_size[0], flatten_size)
        mean, var = self.bottle_neck(x)
        delta = 1e-8
        KL = 0.5 * torch.sum(1 + var - mean**2 - torch.exp(var+delta))
        z = self.sample_z(mean, var, device)
        x = self.fc_z(z)
        ###############
        # unflatten
        x = x.view(middle_size)
        # decoder
        y = self.decoder(x)
        y = y.transpose(1, 3) # (batch_size, melbins, time_steps, 1)
        # calc loss
        reconstruction = F.mse_loss(y, input_spec)
        lower_bound = [KL, reconstruction]
        loss = reconstruction - KL
        output_dict = {'loss':loss, 'x':input_spec, 'y':y}

        return output_dict