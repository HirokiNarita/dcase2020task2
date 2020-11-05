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

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=100):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
            #dropout=0.5
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        #print(x.shape)
        #x = x.reshape((batch_size, self.seq_len, self.n_features))
        plt.figure(figsize=(10,5))
        plt.imshow(x[0,:,:].to('cpu').detach().numpy().T, aspect='auto')
        plt.show()
        print(x.shape)
        x, (hidden_n, cell_n) = self.rnn1(x)
        
        plt.figure(figsize=(10,5))
        #plt.imshow(x[:,0,:].to('cpu').detach().numpy().T, aspect='auto')
        plt.show()
        print(x.shape)
        plt.figure(figsize=(10,5))
        #plt.imshow(hidden_n.to('cpu').detach().numpy().T, aspect='auto')
        plt.show()
        print(hidden_n.shape)
        plt.figure(figsize=(10,5))
        #plt.imshow(cell_n.to('cpu').detach().numpy().T, aspect='auto')
        plt.show()
        print(cell_n.shape)
        x, (hidden_n, _) = self.rnn2(x)
        x = torch.squeeze(x, 1)
        hidden_n = torch.squeeze(hidden_n, 0)#hidden_n.reshape((batch_size, self.embedding_dim))
        return hidden_n

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=100):
        super(Decoder, self).__init__()
        self.seq_len, self.embedding_dim = seq_len, embedding_dim
        self.hidden_dim, self.n_features = 2 * embedding_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
            #dropout=0.5
        )

        self.rnn2 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)
        
    def forward(self, x):
        batch_size = x.shape[0]
        #x = x.repeat(self.seq_len, 1)
        #x = x.permute((1,0,2))
        x = x.reshape(batch_size, self.seq_len, self.embedding_dim)
        #print(x.shape)
        #x = x.repeat(self.batch_size , self.seq_len, self.n_features)
        #x = x.reshape((self.batch_size, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        #x = x.reshape((self.batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)

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
        
        self.encoder = Encoder(seq_len=128, n_features=mel_bins, embedding_dim=128)
        self.decoder = Decoder(seq_len=128, n_features=mel_bins, embedding_dim=128)
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

        x = self.encoder(x)
        x = self.decoder(x) # batch_size, melbins, time_steps
        #y = y.transpose(1, 3) # (batch_size, melbins, time_steps, 1)
        loss = F.mse_loss(x, input_spec)
        
        output_dict = {'loss':loss, 'x':input_spec, 'y':x}

        return output_dict
