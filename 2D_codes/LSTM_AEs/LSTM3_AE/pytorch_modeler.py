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
from sklearn import metrics

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
def calc_auc(y_true, y_pred):
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=config["etc"]["max_fpr"])
    #logger.info("AUC : {}".format(auc))
    #logger.info("pAUC : {}".format(p_auc))
    return auc, p_auc

# training function
def train_net(net, dataloaders_dict, criterion, optimizer, num_epochs, writer):
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)

    tr_epoch_losses = []
    reconstruct_img = defaultdict(list)
    epoch_valid_score = defaultdict(list)
    # epochループ開始
    for epoch in range(num_epochs):
        # loss
        tr_losses = 0
        #losses['valid'] = 0
        labels = []

        # epochごとの訓練と検証のループ
        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
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
                            tr_losses += loss.item()
                tr_epoch_losses.append(tr_losses / len(dataloaders_dict['train']))

            else:
                net.eval()
                preds = np.zeros(len(dataloaders_dict[phase].dataset))
                labels = np.zeros(len(dataloaders_dict[phase].dataset))
                for idx, sample in enumerate(tqdm(dataloaders_dict[phase].dataset)):
                    input = sample['feature']
                    input = torch.unsqueeze(input,0)
                    label = sample['label']
                    input = input.to(device)
                    # optimizerを初期化
                    optimizer.zero_grad()
                    # 順伝播(forward)
                    with torch.no_grad():
                        output_dict = net(input, device)    # (batch_size,input(2D)) 
                        x, y, loss = output_dict['x'], output_dict['y'], output_dict['loss']
                        labels[idx] = np.int64(label.item())
                        preds[idx] = loss.to('cpu').detach().numpy().copy()
                valid_AUC, valid_pAUC = calc_auc(labels, preds)
                epoch_valid_score['AUC'].append(valid_AUC)
                epoch_valid_score['pAUC'].append(valid_pAUC)
                
                if ((epoch+1) % 10 == 0) or (epoch == 0):
                    plt.imshow(y[0,:,:].to('cpu').detach().numpy(), aspect='auto')
                    plt.show()
                    reconstruct_img['input'].append(x)
                    reconstruct_img['output'].append(y)
                    reconstruct_img['label'].append(label)
            # データローダーからminibatchを取り出すループ
        #epoch_losses['valid'].append(losses['valid'] / len(dataloaders_dict['valid']))
        
        logger.info('Epoch {}/{}:train_loss:{:.6f}, valid_AUC:{:.6f}, valid_pAUC:{:.6f}'.format(epoch+1,
                                                                                                num_epochs,
                                                                                                tr_epoch_losses[-1],
                                                                                                epoch_valid_score['AUC'][-1],
                                                                                                epoch_valid_score['pAUC'][-1]))
    
    return {'train_epoch_score':tr_epoch_losses, 'valid_epoch_score':epoch_valid_score, 'reconstruct_img':reconstruct_img, 'model':net}
