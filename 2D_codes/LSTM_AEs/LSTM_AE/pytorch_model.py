""" AutoEncoder in Pytorch. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import matplotlib.pyplot as plt
from pytorch_utils import do_mixup, interpolate, pad_framewise_output
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
                              kernel_size=(5, 5), stride=(2, 2),
                              padding=(2, 2),
                              bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='max'):
        
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
                                        kernel_size=(5, 5),
                                        stride=(2, 2),
                                        padding=(2, 2),
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

class LSTM_AE(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=128, bottom_dim=512):
        super(LSTM_AE, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim, self.bottom_dim = hidden_dim, bottom_dim
        self.lstm_out_features = 2*hidden_dim    #(forward+backward)
        self.input_shape = int(self.n_features * self.seq_len)
        #self.K, self.P, self.S = 5, 2, 1
        
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.bn1 = nn.BatchNorm2d(self.seq_len)

        self.conv1 = ConvBlock5x5(in_channels=2, out_channels=64)
        self.conv2 = ConvBlock5x5(in_channels=64, out_channels=128)
        #self.deconv1 = deConvBlock5x5(in_channels=64, out_channels=2)
        #self.conv_out_features = int((self.n_features - self.K + 2*self.P) / self.S) +1
        self.bn2 = nn.BatchNorm1d(128*8*8)
        self.fc1 = nn.Linear(in_features=128*8*8,
                             out_features=self.bottom_dim)
        self.bn3 = nn.BatchNorm1d(self.bottom_dim)
        self.fc2 = nn.Linear(self.bottom_dim, self.bottom_dim)
        self.fc3 = nn.Linear(self.bottom_dim, self.input_shape)
        #self.bn3 = nn.BatchNorm1d(self.bottom_dim)

        #self.fc2 = nn.Linear(in_features=self.bottom_dim, out_features = self.input_shape)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x, (_, _) = self.rnn1(x)    # (batch_size, seq_len, self.n_features*2)
        x = x.view(batch_size, self.seq_len, 2, self.hidden_dim) # (batch_size, seq_len, channel, hidden_dim)
        x = self.bn1(x) # norm -> seq direction
        x = x.transpose(1, 2)           # (batch_size, seq_len, channel, hidden_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        #print(x.shape)
        #x = self.deconv1(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.bn2(x)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = F.relu(self.fc2(x))
        #x = self.bn3(x)
        x = self.fc3(x)
        x = x.view(batch_size, self.seq_len, self.n_features)
        #x = torch.mean(x, dim=1)
        #x = torch.squeeze(x, dim=3)
        #x = self.bn2(x)
        #x, (_,_) = self.rnn2(x)
        #x = x.view(batch_size, self.seq_len, 2, self.hidden_dim)
        #x = torch.mean(x, dim=2)

        #x, (hidden_n, _) = self.rnn2(x)32768
        
        return x



class LSTM_AutoEncoder(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        
        super(LSTM_AutoEncoder, self).__init__()

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

        self.bn0 = nn.BatchNorm1d(128)
        
        self.lstm_ae = LSTM_AE(seq_len=128, n_features=mel_bins, hidden_dim=128, bottom_dim=1024)
        #self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input, device='cuda:0', mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        #print(x.shape)
        x = torch.squeeze(x, 1)         # (batch_size, time_steps, mel_bins)
        #print(x.shape)
        #print(x.shape)
        #print(x.shape)
        x = x[:, :128, :]   # trim
        input_spec = x        # save input melspectrogram
        #print(x.shape)
        #print()
        x = self.bn0(x)
        
        #if self.training:
        #    x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        #if self.training and mixup_lambda is not None:
        #    x = do_mixup(x, mixup_lambda)

        x = self.lstm_ae(x)
        loss = F.mse_loss(x, input_spec)
        
        output_dict = {'loss':loss, 'x':input_spec, 'y':x}

        return output_dict
