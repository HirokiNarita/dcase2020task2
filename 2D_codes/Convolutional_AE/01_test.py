"""
 @file   01_test.py
 @brief  Script for test
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""
########################################################################
# import default python-library
########################################################################
import os
import glob
import csv
import re
import itertools
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
from sklearn import metrics
import common as com
import pytorch_modeler as modeler
from pytorch_model import VAE as ae
import torch.utils.data
import yaml
yaml.warnings({'YAMLLoadWarning': False})
########################################################################



########################################################################
# load config
########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)
########################################################################
# Setting seed
########################################################################
modeler.set_seed(42)
########################################################################
# Setting I/O path
########################################################################
# input dirs
INPUT_ROOT = config['IO_OPTION']['INPUT_ROOT']
dev_path = INPUT_ROOT + "/dev_data"
add_dev_path = INPUT_ROOT + "/add_dev_data"
eval_path = INPUT_ROOT + "/eval_test"
MODEL_DIR = config['IO_OPTION']['OUTPUT_ROOT'] + '/models'
# machine type
MACHINE_TYPE = config['IO_OPTION']['MACHINE_TYPE']
machine_types = os.listdir(dev_path)
# output dirs
OUTPUT_ROOT = config['IO_OPTION']['OUTPUT_ROOT']
RESULT_DIR = config['IO_OPTION']['OUTPUT_ROOT'] + '/result'
os.makedirs(MODEL_DIR, exist_ok=True)
########################################################################
# for original function
########################################################################
param = {}
param["dev_directory"] = dev_path
param["eval_directory"] = eval_path
param["model_directory"] = MODEL_DIR
param["result_directory"] = RESULT_DIR
param["result_file"] = 'result.csv'
########################################################################
# def
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    com.logger.info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make output result directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        com.logger.info("===========================")
        com.logger.info("[{idx}/{total}] {dirname}".format(
            dirname=target_dir, idx=idx+1, total=len(dirs)))

        machine_type = os.path.split(target_dir)[1]

        com.logger.info("============== MODEL LOAD ==============")

        model_file = "{model}/{machine_type}_model.pth".format(
            model=param["model_directory"],
            machine_type=machine_type)

        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)

        # define AE model
        model = ae().to(device)
        model.eval()
        model.load_state_dict(torch.load(model_file))

        if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

        machine_id_list = get_machine_id_list_for_test(target_dir)

        for id_str in machine_id_list:

            # load list of test files
            test_files, y_true = test_file_list_generator(target_dir, id_str)

            # setup anomaly score file path
            anomaly_score_csv = \
                "{result}/anomaly_score_{machine_type}_{id_str}.csv"\
                .format(result=param["result_directory"],
                        machine_type=machine_type,
                        id_str=id_str)
            anomaly_score_list = []

            com.logger.info(
                "============== BEGIN TEST FOR A MACHINE ID ==============")

            y_pred = [0. for k in test_files]

            for file_idx, file_path in enumerate(test_files):
                try:
                    data = com.file_to_vector_array(
                        file_path,
                        n_mels=config["mel_spectrogram_param"]["n_mels"],
                        frames=config["mel_spectrogram_param"]["frames"],
                        n_fft=config["mel_spectrogram_param"]["n_fft"],
                        hop_length=config["mel_spectrogram_param"]["hop_length"],
                        power=config["mel_spectrogram_param"]["power"])

                    # reconstruction through auto encoder in pytorch
                    feed_data = torch.as_tensor(
                        data, device=device, dtype=torch.float32)
                    with torch.no_grad():
                        _, _, _, y = model(feed_data, device)
                        pred = y
                        pred = pred.to('cpu').detach().numpy().copy()
                        #print(pred)

                    errors = numpy.mean(numpy.square(data - pred), axis=1)
                    y_pred[file_idx] = numpy.mean(errors)
                    anomaly_score_list.append(
                        [os.path.basename(file_path), y_pred[file_idx]])
                except FileNotFoundError:
                    com.logger.error("file broken!!: {}".format(file_path))

            # save anomaly score
            save_csv(save_file_path=anomaly_score_csv,
                     save_data=anomaly_score_list)
            com.logger.info(
                "anomaly score result ->  {}".format(anomaly_score_csv))

            if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(
                    y_true, y_pred, max_fpr=config["etc"]["max_fpr"])
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc))

            com.logger.info(
                "============ END OF TEST FOR A MACHINE ID ============")

        if mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = numpy.mean(
                numpy.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([])

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(
            result=param["result_directory"],
            file_name=param["result_file"])
        com.logger.info("AUC and pAUC results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)

