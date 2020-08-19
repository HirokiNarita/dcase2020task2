############################################################################
# python default library
############################################################################
import os
import glob
import sys
import random
############################################################################
############################################################################
# additional library
############################################################################
# general analysis tool-kit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# sound analysis tool-kit
import librosa
import librosa.core
import librosa.feature

# pytorch
import torch
import torch.utils.data as data
from torch import optim, nn
from torch.utils.data.dataset import Subset

# deeplearning tool-kit
from torchvision import transforms

# etc
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from tqdm import tqdm
import mlflow
############################################################################
# original library
############################################################################
import common as com
import preprocessing as prep
from pytorch_model import AutoEncoder
############################################################################
# Setting seed
############################################################################
com.set_seed(42)
############################################################################
# Setting I/O path
############################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)
# input dirs
INPUT_ROOT = config['IO_OPTION']['INPUT_ROOT']
dev_path = INPUT_ROOT + "/dev_data"
add_dev_path = INPUT_ROOT + "/add_dev_data"
eval_test_path = INPUT_ROOT + "/eval_test"
# machine type
machine_types = os.listdir(dev_path)
# output dirs
OUTPUT_ROOT = config['IO_OPTION']['OUTPUT_ROOT']
############################################################################
# train/valid split
############################################################################
dev_train_paths = {}
add_train_paths = {}

for machine_type in machine_types:
    # dev train
    dev_train_all_paths = ["{}/{}/train/".format(dev_path, machine_type) + file for file in os.listdir("{}/{}/train".format(dev_path, machine_type))]
    dev_train_all_paths = sorted(dev_train_all_paths)
    dev_train_paths[machine_type] = {}
    dev_train_paths[machine_type]['train'], \
    dev_train_paths[machine_type]['valid'] = train_test_split(dev_train_all_paths,
                                                              test_size=config['etc']['test_size'],
                                                              shuffle=False,
                                                             )
    # add_dev train
    add_train_all_paths = ["{}/{}/train/".format(add_dev_path, machine_type) + file for file in os.listdir("{}/{}/train".format(add_dev_path, machine_type))]
    add_train_all_paths = sorted(add_train_all_paths)
    add_train_paths[machine_type] = {}
    add_train_paths[machine_type]['train'], \
    add_train_paths[machine_type]['valid'] = train_test_split(add_train_all_paths,
                                                              test_size=config['etc']['test_size'],
                                                              shuffle=False,
                                                             )
############################################################################
# Make Dataloader
############################################################################
transform = transforms.Compose([
    prep.Wav_to_Melspectrogram(),
    prep.ToTensor()
])
train_dataset = prep.DCASE_task2_Dataset(dev_train_paths[machine_types[0]]['train'], transform=transform)
valid_dataset = prep.DCASE_task2_Dataset(dev_train_paths[machine_types[0]]['valid'], transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=config['fit']['batch_size'],
    shuffle=config['fit']['shuffle'],
    )

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=config['fit']['batch_size'],
    shuffle=False,
    )

dataloaders_dict = {"train": train_loader, "valid": valid_loader}
#############################################################################
# training
#############################################################################

# parameter setting
net = AutoEncoder()
optimizer = optim.Adam(net.parameters())
criterion = nn.MSELoss()

# training function
def train_net(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    # ネットワークがある程度固定であれば、高速化できる
    torch.backends.cudnn.benchmark = True
    # lossの初期化
    epoch_train_loss = []
    epoch_valid_loss = []
    # epochループ開始
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            anomaly_score = {'train':0.0, 'valid':0.0}
            # データローダーからminibatchを取り出すループ
            for sample in tqdm(dataloaders_dict[phase]):
                features = sample['features']
                # サンプル一つ分でのloss
                sample_loss = {'train':0.0, 'valid':0.0}
                # フレームごとに学習させていくループ
                #print(features)
                for row in range(features.shape[0]):
                    # minibatchからフレームごとに取り出す
                    x = features[row,:]
                    # optimizerの初期化
                    optimizer.zero_grad()
                    # 順伝播(forward)
                    with torch.set_grad_enabled(phase == 'train'):
                        x = x.to(device, dtype=torch.float32)
                        outputs = net(x)
                        loss = criterion(outputs, x)    # 再構成誤差
                        preds = outputs                 # 推定値
                        # 訓練時は逆伝播(backforward)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # lossを追加
                    sample_loss[phase] += loss.item()
                # anomaly score
                anomaly_score[phase] += sample_loss[phase] / features.shape[0]
                
            # epoch loss
            if phase == 'train':
                epoch_train_loss.append(anomaly_score[phase] / dataloaders_dict[phase].batch_size)
            else:
                epoch_valid_loss.append(anomaly_score[phase] / dataloaders_dict[phase].batch_size)
                print('-------------')
                print('Epoch {}/{}:train_loss:{:.6f}, valid_loss:{:.6f}'.format(epoch+1, num_epochs, epoch_train_loss[-1], epoch_valid_loss[-1]))

    return {'train_loss':epoch_train_loss, 'valid_loss':epoch_valid_loss}

history = train_net(net, dataloaders_dict, criterion, optimizer, num_epochs=10)