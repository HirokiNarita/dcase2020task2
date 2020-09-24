############################################################################
# load library
############################################################################
# python default library
import os
import shutil
import datetime

# general analysis tool-kit
import numpy as np
from sklearn.model_selection import train_test_split

# pytorch
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

# etc
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import mlflow
from collections import defaultdict

# original library
import common as com
import pytorch_modeler as modeler
from pytorch_model import VAE
############################################################################
# load config and set logger
############################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

log_folder = config['IO_OPTION']['OUTPUT_ROOT']+'/{0}.log'.format(datetime.date.today())
logger = com.setup_logger(log_folder, '00_train.py')
############################################################################
# Setting seed
############################################################################
modeler.set_seed(42)
############################################################################
# Setting I/O path
############################################################################
# input dirs
INPUT_ROOT = config['IO_OPTION']['INPUT_ROOT']
dev_path = INPUT_ROOT + "/dev_data"
add_dev_path = INPUT_ROOT + "/add_dev_data"
# machine type
MACHINE_TYPE = config['IO_OPTION']['MACHINE_TYPE']
machine_types = os.listdir(dev_path)
# output dirs
OUTPUT_ROOT = config['IO_OPTION']['OUTPUT_ROOT']
MODEL_DIR = config['IO_OPTION']['OUTPUT_ROOT'] + '/models'
TB_DIR = config['IO_OPTION']['OUTPUT_ROOT'] + '/tb'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

# copy config
shutil.copy('./config.yaml', OUTPUT_ROOT)

############################################################################
# make path set and train/valid split
############################################################################
'''
train_paths[machine_type]['train' or 'valid'] = path
'''
dev_train_paths = {}
add_train_paths = {}
train_paths = {}

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
    train_paths[machine_type] = {}
    train_paths[machine_type]['train'] = dev_train_paths[machine_type]['train'] + add_train_paths[machine_type]['train']
    train_paths[machine_type]['valid'] = dev_train_paths[machine_type]['valid'] + add_train_paths[machine_type]['valid']

#############################################################################
# run
#############################################################################
def run(machine_type):
    com.tic()
    logger.info('TARGET MACHINE_TYPE: {0}'.format(machine_type))
    logger.info('MAKE DATA_LOADER')
    # dev_train_paths
    dataloaders_dict = modeler.make_dataloader_array(train_paths, machine_type)
    # define writer for tensorbord
    os.makedirs(TB_DIR+'/'+machine_type, exist_ok=True)         # debug
    tb_log_dir = TB_DIR + '/' + machine_type
    writer = SummaryWriter(log_dir = tb_log_dir)
    logger.info('TRAINING')
    # parameter setting
    net = VAE()
    optimizer = optim.Adam(net.parameters())
    criterion = nn.MSELoss()
    num_epochs = config['fit']['num_epochs']
    history = modeler.train_net(net, dataloaders_dict, criterion, optimizer, num_epochs, writer)
    # output
    model = history['model']
    model_out_path = MODEL_DIR+'/{}_model.pth'.format(machine_type)
    torch.save(model.state_dict(), model_out_path)
    logger.info('\n success:{0} \n'.format(machine_type) + \
                    'model_out_path ==> \n {0}'.format(model_out_path))
    #  close writer for tensorbord
    writer.close()
    #modeler.mlflow_log(history, config, machine_type, model_out_path, tb_log_dir)
    com.toc()


if MACHINE_TYPE == 'run_all':
    for machine_type in machine_types:
        run(machine_type)
else:
    machine_type = MACHINE_TYPE
    run(machine_type)
