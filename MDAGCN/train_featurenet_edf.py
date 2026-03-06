import os
import numpy as np
import argparse
import shutil
import gc
import time
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.FeatureNet_EDF import FeatureNet
from model.Utils import *
from model.Dataset import SimpleDataset
from model.DataGenerator_EDF import SleepEDFGenerator

print(128 * '#')
print('Start to train SleepEDF dataset (4 channels)')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str, default='./config/SleepEDF.config')
args = parser.parse_args()
Path, cfgFeature, cfgTrain, cfgModel = ReadConfig(args.c)

channels = int(cfgFeature["channels"])
fold = int(cfgFeature["fold"])
num_epochs_f = int(cfgFeature["epoch_f"])
batch_size_f = int(cfgFeature["batch_size_f"])
optimizer_f = cfgFeature["optimizer_f"]
learn_rate_f = float(cfgFeature["learn_rate_f"])

l2 = float(cfgModel["l2"])

if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save'] + "last.config")

# Read SleepEDF data
print("Loading SleepEDF data...")
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num, Fold_Data, Fold_Label = ReadList['Fold_len'], ReadList['Fold_data'], ReadList['Fold_label']
DataGenerator = SleepEDFGenerator(Fold_Data, Fold_Label)

print(f"SleepEDF data loaded: {len(Fold_Data)} folds, {channels} channels")
print(f"Sample shape: {Fold_Data[0].shape}")


def train_sleepedf_fold(i):
    """
    Train FeatureNet for one fold of SleepEDF dataset.

    Args:
        i (int): Fold index (0-4)
    """
    print(128 * '_')
    print(f'SleepEDF Fold #{i}')

    train_data, train_labels, val_data, val_labels = DataGenerator.getFold(i)

    if train_labels.ndim > 1:
        train_labels = np.argmax(train_labels, axis=1)
    if val_labels.ndim > 1:
        val_labels = np.argmax(val_labels, axis=1)

    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")

    # Create datasets and dataloaders
    trDataset = SimpleDataset(train_data, train_labels)
    cvDataset = SimpleDataset(val_data, val_labels)

    trGen = DataLoader(trDataset, batch_size=batch_size_f, shuffle=True, num_workers=0)
    cvGen = DataLoader(cvDataset, batch_size=batch_size_f, shuffle=False, num_workers=0)

    # Train FeatureNet
    print("Training FeatureNet...")
    model_f = FeatureNet(channels=channels)
    model_f = model_f.cuda()

    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.cuda()

    opt_f = Instantiation_optim(optimizer_f, learn_rate_f, model_f, l2)
    scheduler_f = ReduceLROnPlateau(opt_f, 'min', patience=5, factor=0.5)

    best_acc_f = 0
    best_loss_f = float('inf')
    count_epoch = 0
    history = {'tr_acc': [], 'tr_loss': [], 'va_acc': [], 'va_loss': []}

    for epoch in range(num_epochs_f):
        t_start = time.time()
        model_f.train()
        tr_acc, tr_loss = train_epoch(model_f, trGen, loss_func, opt_f, epoch, accumulation_steps=2)
        va_acc, va_loss = val(model_f, cvGen, loss_func, epoch, False)

        scheduler_f.step(va_loss)

        history['tr_acc'].append(tr_acc)
        history['tr_loss'].append(tr_loss)
        history['va_acc'].append(va_acc)
        history['va_loss'].append(va_loss)

        if va_acc > best_acc_f or va_loss < best_loss_f:
            if va_acc > best_acc_f:
                best_acc_f = va_acc
            if va_loss < best_loss_f:
                best_loss_f = va_loss
            torch.save(model_f.state_dict(), f"{Path['Save']}FeatureNet_Best_{i}.pth")
            print(" Update ", end='')
        else:
            count_epoch += 1
            if count_epoch >= 15:
                print(" EarlyStop ")
                break

        print(f" Time:{time.time() - t_start:.2f}s | LR:{opt_f.param_groups[0]['lr']:.6f}")

    # Extract features
    model_f.load_state_dict(torch.load(f"{Path['Save']}FeatureNet_Best_{i}.pth"))
    model_f.eval()

    train_feature = get_feature_dataset(model_f, trDataset, batch_size_f)
    val_feature = get_feature_dataset(model_f, cvDataset, batch_size_f)

    # Save features
    np.savez(f"{Path['Save']}Feature_{i}.npz",
             train_feature=train_feature, train_targets=np.eye(5)[train_labels],
             val_feature=val_feature, val_targets=np.eye(5)[val_labels])
    saveFile = open(Path['Save'] + "Result_FeatureNet.txt", 'a+')
    print('Fold #' + str(i), file=saveFile)
    print('TR_ACC:', history['tr_acc'], '; TR_loss:', history['tr_loss'],
          '; VA_ACC:', history['va_acc'], '; VA_loss:', history['va_loss'], ';',
          file=saveFile)
    saveFile.close()

    print(f"Features saved for fold {i}")
    return history['tr_loss'], history['tr_acc'], history['va_loss'], history['va_acc']


# Train FeatureNet for each fold
for i in range(fold):
    l_tr, a_tr, l_va, a_va = train_sleepedf_fold(i)
    VariationCurve(a_tr, a_va, f'Acc_Fold_{i}', Path['Save'])
    VariationCurve(l_tr, l_va, f'Loss_Fold_{i}', Path['Save'])

print('SleepEDF FeatureNet training completed!')
