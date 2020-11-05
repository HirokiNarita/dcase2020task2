""" AutoEncoder in Pytorch. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import matplotlib.pyplot as plt
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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
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

class LSTM_AE(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=128, bottom_dim=512):
        super(LSTM_AE, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim, self.bottom_dim = hidden_dim, bottom_dim
        #self.num_direction, self.hidden_dim = num_direction, 2 * n_features
        self.input_shape = int(self.n_features * self.seq_len)
        self.fc1_in_features = int(self.n_features * self.hidden_dim)
        
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.conv1 = ConvBlock(in_channels=1, out_channels=64)
        self.fc1 = nn.Linear(in_features = self.fc1_in_features,
                             out_features = self.bottom_dim)

        self.bn2 = nn.BatchNorm1d(self.bottom_dim)

        self.fc2 = nn.Linear(in_features=self.bottom_dim, out_features = self.input_shape)
    
    def forward(self, x):
        batch_size = x.shape[0]
        #print(x.shape)
        #x = x.reshape((batch_size, self.seq_len, self.n_features))
        print(x.shape)
        x, (hidden_n, cell_n) = self.rnn1(x)    # (batch_size, seq_len, self.n_features*2)
        print(x.shape)
        #x = torch.flatten(x, start_dim=1, end_dim=2)
        x = torch.unsqueeze(x, 1)   # (batch_size, 1, n_features, seq_len)
        print(x.shape)
        x = x.transpose(1, 3)
        print(x.shape)
        ###
        # norm-> seq direction
        x = self.bn1(x)
        x = x.transpose(1, 3)
        ###
        x = self.conv1(x)
        print(x.shape)
        print(jk)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        #x, (hidden_n, _) = self.rnn2(x)
        
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
        
        self.lstm_ae = LSTM_AE(seq_len=128, n_features=mel_bins, hidden_dim=64, bottom_dim=512)
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
