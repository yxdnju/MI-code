import os
import numpy as np
import argparse
import shutil
import gc
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.FeatureNet import *
from model.DataGenerator import kFoldGenerator_train
from model.Utils import *
from model.Dataset import SimpleDataset

# Parse arguments
print(128 * '#')
print('Start to train FeatureNet')

parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str, default='./config/ISRUC_S3.config')
args = parser.parse_args()
Path, cfgFeature, _, _ = ReadConfig(args.c)

channels = int(cfgFeature["channels"])
fold = int(cfgFeature["fold"])
num_epochs_f = int(cfgFeature["epoch_f"])
batch_size_f = int(cfgFeature["batch_size_f"])
optimizer_f = cfgFeature["optimizer_f"]
learn_rate_f = float(cfgFeature["learn_rate_f"])

if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save'] + "last.config")

# Read data
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num, Fold_Data, Fold_Label = ReadList['Fold_len'], ReadList['Fold_data'], ReadList['Fold_label']
DataGenerator = kFoldGenerator_train(Fold_Data, Fold_Label)


def train_featurenet_fold(i):
    print(128 * '_')
    print(f'Fold #{i} | {time.asctime()}')

    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)

    # Compute class weights
    train_labels = np.argmax(train_targets, axis=1) if train_targets.ndim > 1 else train_targets
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = len(train_labels) / (5.0 * class_counts + 1e-8)
    class_weights = torch.FloatTensor(class_weights / np.sum(class_weights) * 5.0).cuda()

    trDataset = SimpleDataset(np.float32(train_data), train_targets)
    cvDataset = SimpleDataset(np.float32(val_data), val_targets)
    trGen = DataLoader(trDataset, batch_size=batch_size_f, shuffle=True)
    cvGen = DataLoader(cvDataset, batch_size=batch_size_f, shuffle=False)

    model = FeatureNet(channels).cuda()
    loss_func = nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = Instantiation_optim(optimizer_f, learn_rate_f, model, 0)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_acc, best_loss = 0, float('inf')
    count_epoch = 0
    history = {'tr_acc': [], 'tr_loss': [], 'va_acc': [], 'va_loss': []}

    for epoch in range(num_epochs_f):
        t_start = time.time()

        # Train and validate
        tr_acc, tr_loss = train_epoch(model, trGen, loss_func, optimizer, epoch, accumulation_steps=2)
        va_acc, va_loss = val(model, cvGen, loss_func, epoch, False)

        scheduler.step(va_loss)

        history['tr_acc'].append(tr_acc)
        history['tr_loss'].append(tr_loss)
        history['va_acc'].append(va_acc)
        history['va_loss'].append(va_loss)

        if va_acc > best_acc or va_loss < best_loss:
            if va_acc > best_acc: best_acc = va_acc
            if va_loss < best_loss: best_loss = va_loss
            torch.save(model.state_dict(), f"{Path['Save']}FeatureNet_Best_{i}.pth")
            print(" Update ", end='')
            count_epoch = 0
        else:
            count_epoch += 1
            if count_epoch >= 15:
                print(" EarlyStop ")
                break

        print(f" Time:{time.time() - t_start:.2f}s | LR:{optimizer.param_groups[0]['lr']:.6f}")

    # Load best model and extract features
    model.load_state_dict(torch.load(f"{Path['Save']}FeatureNet_Best_{i}.pth"))
    model.eval()

    train_feature = get_feature_dataset(model, trDataset, batch_size_f)
    val_feature = get_feature_dataset(model, cvDataset, batch_size_f)

    np.savez(f"{Path['Save']}Feature_{i}.npz",
             train_feature=train_feature, val_feature=val_feature,
             train_targets=train_targets, val_targets=val_targets)
    saveFile = open(Path['Save'] + "Result_FeatureNet.txt", 'a+')
    print('Fold #' + str(i), file=saveFile)
    print('TR_ACC:', history['tr_acc'], '; TR_loss:', history['tr_loss'],
          '; VA_ACC:', history['va_acc'], '; VA_loss:', history['va_loss'], ';',
          file=saveFile)
    saveFile.close()
    del model, train_targets, val_targets, train_data, val_data
    gc.collect()
    return history['tr_loss'], history['tr_acc'], history['va_loss'], history['va_acc']


# Train FeatureNet for each fold
for i in range(fold):
    l_tr, a_tr, l_va, a_va = train_featurenet_fold(i)
    VariationCurve(a_tr, a_va, f'Acc_Fold_{i}', Path['Save'])
    VariationCurve(l_tr, l_va, f'Loss_Fold_{i}', Path['Save'])

print('End of Training.')