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
def train_net(net, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, writer):
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
        tr_eng = 0
        tr_rec = 0
        tr_covd = 0

        #losses['valid'] = 0
        labels = []

        # epochごとの訓練と検証のループ
        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
                for sample in tqdm(dataloaders_dict[phase]):
                    input = sample['feature']
                    input = to_var(input)
                    net, total_loss, sample_energy, recon_error, cov_diag = dagmm_step(net, input, optimizer, scheduler, device)
                    tr_losses += total_loss.data.item()
                    tr_eng += sample_energy.item()
                    tr_rec += recon_error.item()
                    tr_covd += cov_diag.item()
                
                #tr_epoch_losses.append(tr_losses / len(dataloaders_dict['train']))
                # tb
                writer.add_scalar("tr_loss", tr_losses, epoch+1)
                writer.add_scalar("tr_eng", tr_eng, epoch+1)
                writer.add_scalar("tr_rec", tr_rec, epoch+1)
                writer.add_scalar("tr_covd", tr_covd, epoch+1)
                
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
                #epoch_valid_score['AUC'].append(valid_AUC)
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
        
        logger.info('Epoch {}/{}:train_loss:{:.6f}, tr_rec:{:.6f}, tr_eng:{:.6f}, tr_covd:{:.6f}, val_AUC:{:.6f}, val_pAUC:{:.6f}'. \
                                      format(epoch+1,
                                             num_epochs,
                                             tr_losses,
                                             tr_rec,
                                             tr_eng,
                                             tr_covd,
                                             valid_AUC,
                                             valid_pAUC))
    return {'model':net}

def dagmm_step(net, input_data, optimizer, scheduler, device):
    net.train()
    #optimizer.zero_grad()
    nn_out = net(input_data, device)
    total_loss, sample_energy, recon_error, cov_diag = net.loss_function(nn_out['x'],
                                                                         nn_out['x_hat'],
                                                                         nn_out['z'],
                                                                         nn_out['gamma'],
                                                                         config['fit']['lambda_energy'],
                                                                         config['fit']['lambda_cov_diag'])
    #print(total_loss, sample_energy, recon_error, cov_diag)
    net.zero_grad()
    total_loss = torch.clamp(total_loss, max=1e7)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optimizer.step()
    scheduler.step()
    return net, total_loss, sample_energy, recon_error, cov_diag

# def estimate(net, dataloaders_dict, device='cuda:0'):
#     #print("======================TEST MODE======================")
#     net.eval()
#     #self.data_loader.dataset.mode="train"

#     phase = 'train'
#     N = 0
#     mu_sum = 0
#     cov_sum = 0
#     gamma_sum = 0

#     for it, sample in enumerate(tqdm(dataloaders_dict[phase])):
#         input_data, labels = sample['feature'], sample['label']
#         input_data = to_var(input_data)
#         #print(input_data.shape)
#         nn_out = net(input_data, device)
#         z, gamma = nn_out['z'], nn_out['gamma']
#         #{'x':input_img, 'x_hat':dec, 'z_c':z_c, 'z':z, 'gamma':gamma}
#         phi, mu, cov = net.compute_gmm_params(z, gamma)
        
#         batch_gamma_sum = torch.sum(gamma, dim=0)
        
#         gamma_sum += batch_gamma_sum
#         mu_sum += mu * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
#         cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only
        
#         N += input_data.size(0)
        
#     train_phi = gamma_sum / N
#     train_mu = mu_sum / gamma_sum.unsqueeze(-1)
#     train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

#     print("N:",N)
#     print("phi :\n",train_phi)
#     print("mu :\n",train_mu)
#     print("cov :\n",train_cov)

#     train_energy = []
#     train_labels = []
#     train_z = []
#     for it, sample in enumerate(tqdm(dataloaders_dict[phase])):
#         input_data, labels = sample['feature'], sample['label']
#         input_data = to_var(input_data)
#         nn_out = net(input_data)
#         z, gamma = nn_out['z'], nn_out['gamma']
#         sample_energy, cov_diag = net.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
        
#         train_energy.append(sample_energy.data.cpu().numpy())
#         train_z.append(z.data.cpu().numpy())
#         train_labels.append(labels.numpy())

#     phase = 'test'
#     test_energy = []
#     test_labels = []
#     test_z = []
#     for it, (input_data, labels) in enumerate(tqdm(dataloaders_dict[phase])):
#         input_data = to_var(input_data)
#         nn_out = net(input_data)
#         z, gamma = nn_out['z'], nn_out['gamma']
#         sample_energy, cov_diag = net.compute_energy(z, size_average=False)
#         test_energy.append(sample_energy.data.cpu().numpy())
#         test_z.append(z.data.cpu().numpy())
#         test_labels.append(labels.numpy())


#     test_energy = np.concatenate(test_energy,axis=0)
#     test_z = np.concatenate(test_z,axis=0)
#     test_labels = np.concatenate(test_labels,axis=0)

#     combined_energy = np.concatenate([train_energy, test_energy], axis=0)
#     combined_labels = np.concatenate([train_labels, test_labels], axis=0)

#     thresh = np.percentile(combined_energy, 100 - 20)
#     print("Threshold :", thresh)

#     pred = (test_energy > thresh).astype(int)
#     gt = test_labels.astype(int)

#     from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

#     accuracy = accuracy_score(gt,pred)
#     precision, recall, f_score, support = prf(gt, pred, average='binary')

#     print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
    
#     return accuracy, precision, recall, f_score