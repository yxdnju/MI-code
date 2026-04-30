import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from visdom import Visdom

from model.baseModel import baseModel
import time
import os
import yaml
from data.data_utils import *
from data.dataset import eegDataset
from utils import *
import time
import torch.optim.lr_scheduler as lr_scheduler

torch.set_num_threads(10)
def setRandom(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()

def main(config):
    data_path = config['data_path']
    out_folder = config['out_folder']
    random_folder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    best_acc_list = []

    lr = config['lr']
    # time.sleep(3600 * 7)
    for subId in range(1, 10):

        train_datafile = 'B0' + str(subId) + 'T'
        test_datafile = 'B0' + str(subId) + 'E'

        out_path = os.path.join(out_folder, config['network'], 'sub' + str(subId), random_folder)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        print("Results will be saved in folder: " + out_path)

        dictToYaml(os.path.join(out_path, 'config.yaml'), config)

        setRandom(config['random_seed'])

        train_data, train_labels = load_BCI42_data(data_path, train_datafile)
        test_data, test_labels = load_BCI42_data(data_path, test_datafile)

        train_dataset = eegDataset(train_data, train_labels)
        test_dataset = eegDataset(test_data, test_labels)

        net_args = config['network_args']
        net = eval(config['network'])(**net_args)
        print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

        model = baseModel(net, config, optimizer, loss_func, result_savepath=out_path)

        best_acc = model.train_test(train_dataset, test_dataset, 1, subId)
        best_acc_list.append(best_acc)
        print(f"受试者 {subId} 的最佳准确率: {best_acc:.4f}")
    avg_best_acc = sum(best_acc_list) / len(best_acc_list)
    print(f"\n所有受试者平均最佳准确率: {avg_best_acc:.4f}")


if __name__ == '__main__':
    configFile = 'config/bciiv2b_transnet.yaml'
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)
