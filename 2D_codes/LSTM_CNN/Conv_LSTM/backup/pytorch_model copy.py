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
"""
import torch.nn as nn
nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
"""
"""
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
"""

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


"""
class deConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_layer=False):
        
        super(deConvBlock, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels, 
                                        out_channels=out_channels,
                                        kernel_size=(2, 2),
                                        stride=(2, 2),
                                        bias=False
                                        )
                       
        self.deconv2 = nn.ConvTranspose2d(in_channels=out_channels, 
                                          out_channels=out_channels,
                                          kernel_size=(1, 1),
                                          stride=(1, 1),
                                          bias=False
                                          )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        if out_layer == False:
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight(out_layer)
        
    def init_weight(self,out_layer=False):
        init_layer(self.deconv1)
        init_bn(self.bn1)
        if out_layer == False:
            init_layer(self.deconv2)
            init_bn(self.bn2)

        
    def forward(self, input,  out_layer=False):
        
        x = input
        x = F.relu_(self.bn1(self.deconv1(x)))
        if out_layer == True:
            x = self.deconv2(x)
        else:
            x = F.relu_(self.bn2(self.deconv2(x)))
        
        
        return x
"""
class CNN14PANNsAutoEncoder(nn.Module):
    """
        CNN14AutoEncoder
    """

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        
        super(CNN14PANNsAutoEncoder, self).__init__()

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

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.deconv_block1 = deConvBlock(in_channels=2048, out_channels=1024)
        self.deconv_block2 = deConvBlock(in_channels=1024, out_channels=512)
        self.deconv_block3 = deConvBlock(in_channels=512, out_channels=256)
        self.deconv_block4 = deConvBlock(in_channels=256, out_channels=128)
        self.deconv_block5 = deConvBlock(in_channels=128, out_channels=64)
        self.deconv_block6 = deConvBlock(in_channels=64, out_channels=1, out_layer=True)

        #self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        #init_layer(self.fc1)
        #init_layer(self.fc_audioset)

    def encoder(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def decoder(self, x):
        x = self.deconv_block1(x)
        x = self.deconv_block2(x)
        x = self.deconv_block3(x)
        x = self.deconv_block4(x)
        x = self.deconv_block5(x)
        x = self.deconv_block6(x, out_layer=True)
        return x

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        logmel_spec_x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        logmel_spec_x = logmel_spec_x[:, :, :513,:]
        x = logmel_spec_x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        #if self.training:
        #    x = self.spec_augmenter(x)

        # Mixup on spectrogram
        #if self.training and mixup_lambda is not None:
        #    x = do_mixup(x, mixup_lambda)
        x = self.encoder(x)
        y = self.decoder(x)
        #print('y', y.shape)
        #print('logmel_spec_x', logmel_spec_x.shape)
        loss = 'hoge'#F.mse_loss(y, logmel_spec_x)
        #x = torch.mean(x, dim=3)
        
        #(x1, _) = torch.max(x, dim=2)
        #x2 = torch.mean(x, dim=2)
        #x = x1 + x2
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = F.relu_(self.fc1(x))
        #embedding = F.dropout(x, p=0.5, training=self.training)

        #clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'loss':loss, 'x':x, 'y':y}

        return output_dict
