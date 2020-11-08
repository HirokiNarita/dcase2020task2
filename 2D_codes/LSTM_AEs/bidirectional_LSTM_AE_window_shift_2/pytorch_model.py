""" AutoEncoder in Pytorch. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import matplotlib.pyplot as plt
from pytorch_utils import do_mixup, interpolate, pad_framewise_output, hopping
import numpy as np

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
                              padding=(2, 2),
                              bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = self.bn1(F.relu_(self.conv1(x)))
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
        
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels, 
                                        out_channels=out_channels,
                                        kernel_size=(2, 2),
                                        stride=(2, 2),
                                        #padding=(2, 2),
                                        bias=False
                                        )

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.deconv1)
        init_bn(self.bn1)

        
    def forward(self, input):
        x = input
        x = self.bn1(F.relu_(self.deconv1(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, in_features, hidden_dim, mid_dim=64):
        super(Encoder, self).__init__()
        self.in_features, self.mid_dim = in_features, mid_dim, 
        self.hidden_dim = hidden_dim
        #self.lstm_out_features = 2*hidden_dim    #(forward+backward)
        #self.input_shape = int(self.n_features * self.seq_len)

        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(in_features=in_features*2, out_features=mid_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        #print(x.shape)
        x, _ = self.lstm(x)
        #print(x.shape)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        return x

class Decoder(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(Decoder, self).__init__()
        self.in_features, self.out_features = in_features, out_features, 
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)

        self.fc2 = nn.Linear(out_features, out_features)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(out_features, out_features)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(out_features, out_features)
        #self.fc1 = nn.Linear(2*self.seq_len*self.n_features, self.seq_len*self.n_features)

    def forward(self, x):
        #batch_size = x.shape[0]
        #x = x.view(batch_size, self.seq_len, self.hidden_dim*2) # (batch_size, seq_len, channel, hidden_dim)
        #print(x)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        #x, _ = self.lstm(x)
        x = self.bn1(F.relu(self.fc2(x)))
        x = self.bn2(F.relu(self.fc3(x)))
        x = self.bn3(F.relu(self.fc4(x)))
        #print(x)
        x = self.fc5(x)
        #x = x.view(batch_size, -1,-1) # (batch_size, seq_len, hidden_dim)
        #x = self.fc1(x)
        #x = x.view(batch_size, self.seq_len, self.n_features)
        return x

class LSTM_AutoEncoder(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
                 aug_window_size, aug_hop_size):
        
        super(LSTM_AutoEncoder, self).__init__()
        self.aug_window_size, self.aug_hop_size = aug_window_size, aug_hop_size
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

        self.bn0 = nn.BatchNorm1d(32)

        self.encoder = Encoder(in_features=128, hidden_dim=128, mid_dim=64)
        self.decoder = Decoder(in_features=64, hidden_dim=32, out_features=128)
        #self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input, device='cuda:0', mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = torch.squeeze(x, 1)         # (batch_size, time_steps, mel_bins)
        x = x[:, :128, :]   # trim

        x = hopping(x, device, window_size=self.aug_window_size, hop_size=self.aug_hop_size)
        input_spec = x # save input melspectrogram
        #print(x.shape)
        #print(x)
        x = self.bn0(x)
        #if self.training:
        #    x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        #if self.training and mixup_lambda is not None:
        #    x = do_mixup(x, mixup_lambda)
        x = self.encoder(x)
        x = self.decoder(x)
        loss = F.mse_loss(x, input_spec)
        
        output_dict = {'loss':loss, 'x':input_spec, 'y':x}

        return output_dict
