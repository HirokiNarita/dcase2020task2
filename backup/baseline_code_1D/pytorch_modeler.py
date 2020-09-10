############################################################################
# load library
############################################################################

# python default library
import os
import random

# general analysis tool-kit
import numpy as np

# pytorch
import torch
import torch.utils.data as data
from torch import optim, nn
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter

# deeplearning tool-kit
from torchvision import transforms

# etc
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from tqdm import tqdm
import mlflow
from collections import defaultdict

# original library
import common as com
import preprocessing as prep

############################################################################
# load config
############################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)
############################################################################
# Setting seed
############################################################################
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

############################################################################
# Make Dataloader
############################################################################
def make_dataloader(train_paths, machine_type):
    transform = transforms.Compose([
        prep.Wav_to_Melspectrogram(),
        prep.ToTensor()
    ])
    train_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['train'], transform=transform)
    valid_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid'], transform=transform)

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
    
    return dataloaders_dict
#############################################################################
# mlflow
#############################################################################
def mlflow_log(history, config, machine_type, out_path, tb_log_dir):
    mlflow.set_tracking_uri(config['IO_OPTION']['MLFLOW_PATH']+'/mlruns')
    run_name = config['IO_OPTION']['model_name']+'_'+machine_type
    with mlflow.start_run(run_name=run_name) as run:
        # IO_OPTION and etc into mlflow
        mlflow.set_tags(config['IO_OPTION'])
        mlflow.set_tags(config['etc'])
        mlflow.set_tag('machine_type', machine_type)
        mlflow.set_tag('tb_log_dir', tb_log_dir)
        # Log spectrogram_param into mlflow
        for key, value in config['mel_spectrogram_param'].items():
            mlflow.log_param(key, value)
        # log fit param
        for key, value in config['fit'].items():
            mlflow.log_param(key, value)
        # Log other info
        mlflow.log_param('loss_type', 'MSE')
        
        # Log results into mlflow
        mlflow.log_metric('train_epoch_score', history['train_epoch_score'])
        mlflow.log_metric('valid_epoch_score', history['valid_epoch_score'])

        # Log model
        mlflow.log_artifact(out_path)
    mlflow.end_run()

#############################################################################
# training
#############################################################################

# training function
def train_net(net, dataloaders_dict, criterion, optimizer, num_epochs, writer):
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)
    # count
    total_mse_count = {'train':0, 'valid':0}
    total_score_count = {'train':0, 'valid':0}
    # epoch_score保存用のdict
    epoch_scores = defaultdict(list)
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
                for row in range(features.shape[0]):
                    # minibatchからフレームごとに取り出す
                    x = features[row,:]
                    # optimizerの初期化
                    optimizer.zero_grad()
                    # 順伝播(forward)
                    with torch.set_grad_enabled(phase == 'train'):
                        x = x.to(device, dtype=torch.float32)
                        outputs = net(x)
                        loss = criterion(outputs, x)    # MSE(1/640)
                        preds = outputs                 # decoder output
                        # 訓練時は逆伝播(backforward)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        writer.add_scalar('{}_total_mse'.format(phase), loss.item(), total_mse_count[phase])
                        total_mse_count[phase]+=1
                    # lossを追加
                    sample_loss[phase] += loss.item()
                # anomaly score
                anomaly_score[phase] += sample_loss[phase] / features.shape[0]
                writer.add_scalar('{}_total_score'.format(phase), anomaly_score[phase], total_score_count[phase])
                total_score_count[phase]+=1
            # epoch score
            epoch_score = anomaly_score[phase] / dataloaders_dict[phase].batch_size
            epoch_scores[phase].append(epoch_score)
            writer.add_scalar('{}_epoch_score'.format(phase), epoch_score, epoch)
            
            if phase == 'valid':
                print('-------------')
                print('Epoch {}/{}:train_score:{:.6f}, valid_score:{:.6f}'.format(epoch+1, num_epochs, epoch_scores['train'][-1], epoch_scores['valid'][-1]))
    
    return {'train_epoch_score':epoch_scores['train'][-1], 'valid_epoch_score':epoch_scores['valid'][-1], 'model':net}
