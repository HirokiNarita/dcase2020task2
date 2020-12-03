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
from pytorch_utils import to_var

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
    torch.backends.cudnn.benchmark = True # False
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
        num_workers=2,
        pin_memory=True
        )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config['fit']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
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
def train_net(net, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, writer):
    # make img outdir
    #img_out_dir = IMG_DIR + '/' + machine_type
    #os.makedirs(img_out_dir, exist_ok=True)
    # GPUが使えるならGPUモードに
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    net.to(device)

    #tr_epoch_losses = []
    #reconstruct_img = defaultdict(list)
    #epoch_valid_score = defaultdict(list)
    valid_AUCs = []
    valid_pAUCs = []
    best_AUC = 0
    # epochループ開始
    for epoch in range(num_epochs):
        # loss
        tr_losses = 0
        tr_eng = 0
        tr_rec = 0
        tr_covd = 0
        tr_metaloss = 0

        #losses['valid'] = 0

        # epochごとの訓練と検証のループ
        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
                for sample in tqdm(dataloaders_dict[phase]):
                    input = sample['feature']
                    input = to_var(input)
                    net, total_loss, sample_energy, recon_error, cov_diag, meta_loss = dagmm_step(net, input, optimizer, scheduler, device)
                    tr_losses += total_loss.data.item()
                    tr_eng += sample_energy.item()
                    tr_rec += recon_error.item()
                    tr_covd += cov_diag.item()
                    tr_metaloss += meta_loss.item()
                
                #tr_epoch_losses.append(tr_losses / len(dataloaders_dict['train']))
                # tb
                writer.add_scalar("tr_loss", tr_losses, epoch+1)
                writer.add_scalar("tr_eng", tr_eng, epoch+1)
                writer.add_scalar("tr_rec", tr_rec, epoch+1)
                writer.add_scalar("tr_covd", tr_covd, epoch+1)
                writer.add_scalar("tr_metaloss", tr_metaloss, epoch+1)
                
            else:
                net.eval()
                test_energy = []
                test_labels = []
                test_z = []
                #preds = np.zeros(len(dataloaders_dict[phase].dataset))
                #labels = np.zeros(len(dataloaders_dict[phase].dataset))
                for it, sample in enumerate(tqdm(dataloaders_dict[phase])):
                    input = sample['feature']
                    label = sample['label']
                    input = to_var(input)
                    nn_out = net(input)
                    z, gamma = nn_out['z'], nn_out['gamma']
                    sample_energy, cov_diag = net.compute_energy(z, size_average=False)
                    test_energy.append(sample_energy.data.cpu().numpy())
                    test_z.append(z.data.cpu().numpy())
                    test_labels.append(label.numpy())
                test_energy = np.concatenate(test_energy,axis=0)
                test_z = np.concatenate(test_z,axis=0)
                test_labels = np.concatenate(test_labels,axis=0)
                valid_AUC, valid_pAUC = calc_auc(test_labels, test_energy)
                valid_AUCs.append(valid_AUC)
                valid_pAUCs.append(valid_pAUC)
                #epoch_valid_score['pAUC'].append(valid_pAUC)

                writer.add_scalar("valid_AUC", valid_AUC, epoch+1)
                writer.add_scalar("valid_pAUC", valid_pAUC, epoch+1)
                
                #if ((epoch+1) % 10 == 0) or (epoch == 0):
                #    plt.imshow(y[0,:,:].to('cpu').detach().numpy(), aspect='auto')
                #    plt.show()
                #    reconstruct_img['input'].append(x)
                #    reconstruct_img['output'].append(y)
                #    reconstruct_img['label'].append(label)
            # データローダーからminibatchを取り出すループ
        #epoch_losses['valid'].append(losses['valid'] / len(dataloaders_dict['valid']))
        
        # early stopping
        if best_AUC < valid_AUCs[-1]:
            best_AUC = valid_AUCs[-1]
            best_pAUC = valid_pAUCs[-1]
            best_net = net
            cnt = 0
            best_epoch = epoch
        else:
            cnt+=1
            if cnt == config['fit']['early_stopping']:
                logger.info('Early stopping : best Epoch {}/{}, AUC:{:.6f}, pAUC:{:.6f}'.format(best_epoch+1,
                                                                                                num_epochs,
                                                                                                best_AUC,
                                                                                                best_pAUC))
                return {'model':best_net}

        logger.info('Epoch {}/{}:train_loss:{:.6f}, tr_rec:{:.6f}, tr_eng:{:.6f}, tr_covd:{:.6f}, tr_metaloss:{:.6f}, val_AUC:{:.6f}, val_pAUC:{:.6f}'. \
                                      format(epoch+1,
                                             num_epochs,
                                             tr_losses,
                                             tr_rec,
                                             tr_eng,
                                             tr_covd,
                                             tr_metaloss,
                                             valid_AUC,
                                             valid_pAUC))
    
    logger.info('Early stopping : best Epoch {}/{}, AUC:{:.6f}, pAUC:{:.6f}'.format(best_epoch+1,
                                                                                    num_epochs,
                                                                                    best_AUC,
                                                                                    best_pAUC))
    return {'model':best_net}

def dagmm_step(net, input_data, optimizer, scheduler, device):
    net.train()
    #optimizer.zero_grad()
    nn_out = net(input_data, device)
    total_loss, sample_energy, recon_error, cov_diag = net.loss_function(nn_out['x'],
                                                                         nn_out['x_hat'],
                                                                         nn_out['z'],
                                                                         nn_out['gamma'],
                                                                         nn_out['meta_loss'],
                                                                         config['fit']['lambda_energy'],
                                                                         config['fit']['lambda_cov_diag'])
    #print(total_loss, sample_energy, recon_error, cov_diag)
    net.zero_grad()
    total_loss = torch.clamp(total_loss, max=1e7)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optimizer.step()
    scheduler.step()
    return net, total_loss, sample_energy, recon_error, cov_diag, nn_out['meta_loss']