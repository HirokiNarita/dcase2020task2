############################################################################
# load library
############################################################################

# python default library
import os
import random
import datetime

# general analysis tool-kit
import numpy as np
import matplotlib.pyplot as plt

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
log_folder = config['IO_OPTION']['OUTPUT_ROOT']+'/{0}.log'.format(datetime.date.today())
logger = com.setup_logger(log_folder, 'pytorch_modeler.py')
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
        prep.extract_waveform(),
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
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)

    epoch_losses = defaultdict(list)
    reconstruct_img = defaultdict(list)
    
    # epochループ開始
    for epoch in range(num_epochs):
        # loss
        losses = {}
        losses['train'] = 0
        losses['valid'] = 0

        # epochごとの訓練と検証のループ
        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            # データローダーからminibatchを取り出すループ
            for sample in tqdm(dataloaders_dict[phase]):
                input = sample['feature']
                input = input.to(device)
                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝播(forward)
                with torch.set_grad_enabled(phase == 'train'):
                    output_dict = net(input, device)    # (batch_size,input(2D)) 
                    x, y, loss = output_dict['x'], output_dict['y'], output_dict['loss']
                    # 訓練時はbackprop
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                losses[phase] += loss.item()

        epoch_losses['train'].append(losses['train'] / len(dataloaders_dict['train']))
        epoch_losses['valid'].append(losses['valid'] / len(dataloaders_dict['valid']))
        
        logger.info('Epoch {}/{}:train_loss:{:.6f}, valid_loss:{:.6f}'.format(epoch+1,
                                                                              num_epochs,
                                                                              epoch_losses['train'][-1],
                                                                              epoch_losses['valid'][-1]
                                                                              ))
        
        if ((epoch+1) % 10 == 0) or (epoch == 0):
            reconstruct_img['input'].append(x)
            reconstruct_img['output'].append(y)
    
    return {'train_epoch_score':epoch_losses['train'], 'valid_epoch_score':epoch_losses['valid'], 'reconstruct_img':reconstruct_img, 'model':net}
