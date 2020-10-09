########################################################################
# import python-library
########################################################################
# python library
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
from scipy.stats import zscore
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)


class Wav_to_Melspectrogram(object):
    """
    wavデータロード(波形) -> ログメルスペクトログラム
    
    Attributes
    ----------
    dims = n_mels * frames
    sound_data : numpy.ndarray.shape = (timecourse, dims)
    """
    def __init__(self, sound_data=None):
        self.sound_data = sound_data
    
    def __call__(self, sample):
        self.sound_data = com.file_to_vector_array(
            sample['wav_name'],
            config['mel_spectrogram_param']['n_mels'],
            config['mel_spectrogram_param']['frames'],
            config['mel_spectrogram_param']['n_fft'],
            config['mel_spectrogram_param']['hop_length'],
            config['mel_spectrogram_param']['power']
        )
        self.labels = np.full((self.sound_data.shape[0]), sample['label'])
        
        return {'features': self.sound_data, 'labels': self.labels}


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        features, labels = sample['features'], sample['labels']
        
        return {'features': torch.from_numpy(features), 'labels': torch.from_numpy(labels)}


class DCASE_task2_Dataset_path(torch.utils.data.Dataset):
    '''
    Attribute
    ----------
    
    '''
    
    def __init__(self, file_list, transform=None):
        self.transform = transform
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # ファイル名でlabelを判断
        if "normal" in file_path:
            label = 0
        else:
            label = 1
        
        sample = {'wav_name':file_path, 'label':label}
        sample = self.transform(sample)
        
        return sample

class DCASE_task2_Dataset_array(torch.utils.data.Dataset):
    '''
    Attribute
    ----------
    
    '''
    
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.dataset, self.labels = self.make_dataset(file_list)
        
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        features = self.dataset[idx, :]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        
        sample = {'features':features, 'label':label}

        return sample
    
    def make_dataset(self, file_list):

        for file_num in range(len(file_list)):
            # check label
            if "normal" in file_list[file_num]:
                label = 0
            else:
                label = 1

            # load logmel features
            array_2d = com.file_to_vector_array(
                file_list[file_num],
                config['mel_spectrogram_param']['n_mels'],
                config['mel_spectrogram_param']['frames'],
                config['mel_spectrogram_param']['n_fft'],
                config['mel_spectrogram_param']['hop_length'],
                config['mel_spectrogram_param']['power']
            )
            if file_num == 0:
                dataset = np.zeros(
                    (array_2d.shape[0] * len(file_list), array_2d.shape[1]), float)
                labels = np.zeros( (array_2d.shape[0] * len(file_list) ),int)
            
            dataset[array_2d.shape[0] * file_num:
                    array_2d.shape[0] * (file_num + 1), :] = array_2d
            
            labels[array_2d.shape[0] * file_num:
                   array_2d.shape[0] * (file_num + 1)] = np.full((array_2d.shape[0]), label)
        
        #dataset = zscore(dataset, axis=0)
        
        return torch.from_numpy(dataset).float(), torch.from_numpy(labels)


        