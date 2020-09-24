########################################################################
# import python-library
########################################################################
# python library
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
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
        
        return {'features': torch.from_numpy(features).float(), 'labels': torch.from_numpy(labels)}


class DCASE_task2_Dataset(torch.utils.data.Dataset):
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
