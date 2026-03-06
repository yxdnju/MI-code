import os
import numpy as np
import argparse
import shutil
import gc
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import collections

from model.Dataset import SimpleDataset
from model.MDAGCN import MDAGCN
from model.Utils import *

print(128 * '#')
print('Start to evaluate MDAGCN.')



def rename_state_dict_keys(state_dict):
    """
    Replace the keys of the state_dict with the new keys.
    """
    new_state_dict = collections.OrderedDict()

    key_mapping = {
        'gcn_block.': 'gcn.',
        'dense_class.3.': 'dense_class.1.',
        'dense_class.5.': 'dense_class.2.',
    }

    for key, value in state_dict.items():
        new_key = key

        if any(skip_key in new_key for skip_key in ['transformer_encoder.', 'pos_encoder.']):
            continue

        for old_prefix, new_prefix in key_mapping.items():
            if old_prefix in new_key:
                new_key = new_key.replace(old_prefix, new_prefix)
                break

        if 'gcn_block.cnn_GL.convs.0.' in key:
            new_key = key.replace('gcn_block.cnn_GL.convs.0.', 'gcn.cnn_GL.')
        elif 'gcn_block.cnn_GL.convs.1.' in key or 'gcn_block.cnn_GL.convs.2.' in key:
            continue
        elif 'gcn_block.cnn_GL.project.' in key:
            continue

        new_state_dict[new_key] = value

    return new_state_dict


# 1. Get configuration
parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str, help="configuration file", default="config/SleepEDF.config")
parser.add_argument("-g", type=str, help="GPU number to use", default='0')
args = parser.parse_args()
Path, _, cfgTrain, cfgModel = ReadConfig(args.c)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.g

# 2. Parse parameters
channels = int(cfgTrain["channels"])
fold = int(cfgTrain["fold"])
context = int(cfgTrain["context"])
batch_size = int(cfgTrain["batch_size"])

GLalpha = float(cfgModel["GLalpha"])
num_of_chev_filters = int(cfgModel["cheb_filters"])
num_of_time_filters = int(cfgModel["time_filters"])
time_conv_strides = int(cfgModel["time_conv_strides"])
time_conv_kernel = int(cfgModel["time_conv_kernel"])
cheb_k = int(cfgModel["cheb_k"])
dropout = float(cfgModel["dropout"])

# 3. Read data
print("Loading data...")
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num = ReadList['Fold_len']
Fold_Data = ReadList['Fold_data']
Fold_Label = ReadList['Fold_label']

print("Data loaded successfully")
Fold_Num_c = Fold_Num + 1 - context
print('Total samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')

# 4. k-fold cross-validation evaluation
all_scores = []
AllPred = None
AllTrue = None

for i in range(fold):
    print(128 * '_')
    print('Fold #', i)
    print(time.asctime(time.localtime(time.time())))

    Features = np.load(Path['Save'] + 'Feature_' + str(i) + '.npz', allow_pickle=True)
    val_feature = np.float32(Features['val_feature'])
    val_targets = Features['val_targets']

    val_labels = np.argmax(val_targets, axis=1)

    train_feature = np.float32(Features['train_feature'])
    f_mean = np.mean(train_feature, axis=(0, 1), keepdims=True)
    f_std = np.std(train_feature, axis=(0, 1), keepdims=True) + 1e-8
    val_feature = (val_feature - f_mean) / f_std

    val_feature, val_labels = AddContext_SingleSub(val_feature, val_labels, context)
    print('Validation feature shape:', val_feature.shape)

    valDataset = SimpleDataset(val_feature, val_labels)
    valGen = DataLoader(valDataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0)

    model = MDAGCN(context, channels, 128, cheb_k, num_of_chev_filters, num_of_time_filters,
                   time_conv_strides, time_conv_kernel, GLalpha, dropout, num_classes=5)
    model = model.cuda()

    print('Loading model weights...')
    state_dict = torch.load(Path['Save'] + 'MDAGCN_Best_' + str(i) + '.pth')
    new_state_dict = rename_state_dict_keys(state_dict)

    model.load_state_dict(new_state_dict, strict=False)
    print('Model weights loaded successfully')

    print('[TEST] Fold #', i)
    loss_func = nn.CrossEntropyLoss()
    acc, loss, loss_c, test_pred, test_true = val_MDAGCN(model, valGen, loss_func, 0, True)

    all_scores.append(acc)
    print('Fold {} Acc: {:.4f}, Loss: {:.4f}'.format(i, acc, loss))

    if AllPred is None:
        AllPred = test_pred
        AllTrue = test_true
    else:
        AllPred = np.concatenate((AllPred, test_pred))
        AllTrue = np.concatenate((AllTrue, test_true))

    del model, val_feature, val_targets, state_dict, new_state_dict
    gc.collect()

# 5. Final results
print(128 * '=')
print("All folds accuracy: ", [f"{score:.4f}" for score in all_scores])
print("Average accuracy: {:.4f} ± {:.4f}".format(np.mean(all_scores), np.std(all_scores)))

print(128 * '=')
PrintScore(AllTrue, AllPred)
PrintScore(AllTrue, AllPred, savePath=Path['Save'], savefile='Result_MDAGCN_Evaluation.txt')

# Save confusion matrix
ConfusionMatrix(AllTrue, AllPred, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath=Path['Save'])

print('MDAGCN evaluation completed successfully.')
print(128 * '#')
