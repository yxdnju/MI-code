import os
import math
import argparse
import shutil
import gc

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.MDAGCN import MDAGCN
from model.Utils import *

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.Dataset import SimpleDataset

print(128 * '#')
print('Start to train MDAGCN.')

# # 1. Get configuration

# ## 1.1. Read .config file

parser = argparse.ArgumentParser()
parser.add_argument("-c", type = str, help = "configuration file", required = False, 
                    default='./config/SleepEDF.config')
parser.add_argument("-g", type = str, help = "GPU number to use", required = False)
args = parser.parse_args()
Path, _, cfgTrain, cfgModel = ReadConfig('./config/SleepEDF.config')

# ## 1.2. Analytic parameters

# [train] parameters ('_f' means FeatureNet)
channels   = int(cfgTrain["channels"])
fold       = int(cfgTrain["fold"])
context    = int(cfgTrain["context"])
num_epochs = int(cfgTrain["epoch"])
batch_size = int(cfgTrain["batch_size"])
optimizer  = cfgTrain["optimizer"]
learn_rate = float(cfgTrain["learn_rate"])

# [model] parameters

GLalpha               = float(cfgModel["GLalpha"])
num_of_chev_filters   = int(cfgModel["cheb_filters"])
num_of_time_filters   = int(cfgModel["time_filters"])
time_conv_strides     = int(cfgModel["time_conv_strides"])
time_conv_kernel      = int(cfgModel["time_conv_kernel"])
cheb_k                = int(cfgModel["cheb_k"])
l2                    = float(cfgModel["l2"])
dropout               = float(cfgModel["dropout"])

# ## 1.3. Parameter check and enable

# Create save pathand copy .config to it
if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save']+"last.config")


# # 2. Read data and process data
# Each fold corresponds to one subject's data (ISRUC-S3 dataset)
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num   = ReadList['Fold_len']    # Num of samples of each fold

print("Read data successfully")
Fold_Num_c  = Fold_Num + 1 - context
print('Number of samples: ',np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')

# # 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
fit_loss = []
fit_acc = []
fit_val_loss = []
fit_val_acc = []

def train_MDAGCN_fold(i):
    print(128*'_')
    print('Fold #', i)
    print(time.asctime(time.localtime(time.time())))
    
    # get i th-fold data
    Features = np.load(Path['Save']+'Feature_'+str(i)+'.npz', allow_pickle=True)
    train_feature = np.float32(Features['train_feature'])
    train_targets = Features['train_targets']
    val_feature   = np.float32(Features['val_feature'])
    val_targets   = Features['val_targets']

    # Calculate class weights
    if train_targets.ndim > 1:
        train_labels = np.argmax(train_targets, axis=1)
    else:
        train_labels = train_targets
    if val_targets.ndim > 1:
        val_labels = np.argmax(val_targets, axis=1)
    else:
        val_labels = val_targets

    class_counts = np.bincount(train_labels, minlength=5)
    total_samples = len(train_labels)

    class_weights = np.sqrt(total_samples / (class_counts + 1e-8))
    class_weights = class_weights / np.sum(class_weights) * 5.0


    # Feature normalization
    f_mean = np.mean(train_feature, axis=(0, 1), keepdims=True)
    f_std = np.std(train_feature, axis=(0, 1), keepdims=True) + 1e-8

    train_feature = (train_feature - f_mean) / f_std
    val_feature = (val_feature - f_mean) / f_std
    ## Use the feature to train MDAGCN
    print('Feature',train_feature.shape,val_feature.shape)
    train_feature, train_labels  = AddContext_MultiSub(train_feature, train_labels,
                                                        np.delete(Fold_Num.copy(), [i,(i+9)%10]), context, i)
    # train_feature, train_labels  = AddContext_MultiSub_EDF(train_feature, train_labels,
    #                                                     np.delete(Fold_Num.copy(), [i,(i+9)%10]), context, i)
    val_feature, val_labels      = AddContext_SingleSub(val_feature, val_labels, context)
    print('Feature with context:',train_feature.shape, val_feature.shape)
    
    trDataset = SimpleDataset(np.float32(train_feature), train_labels)
    cvDataset = SimpleDataset(np.float32(val_feature), val_labels)
    trGen = DataLoader(trDataset,
                       batch_size = batch_size,
                       shuffle = True,
                       num_workers = 0)
    cvGen = DataLoader(cvDataset,
                       batch_size = batch_size,
                       shuffle = False,
                       num_workers = 0)
    
    ## build MDAGCN & train
    model = MDAGCN(context, channels, 128, cheb_k, num_of_chev_filters, num_of_time_filters,
                   time_conv_strides, time_conv_kernel, GLalpha, dropout, num_classes=5)
    model = model.cuda()

    loss_func = nn.CrossEntropyLoss(weight = torch.FloatTensor(class_weights))
    loss_func = loss_func.cuda()
    
    opt = Instantiation_optim(optimizer, learn_rate, model, l2) # optimizer of FeatureNet

    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    # Gradient clipping
    max_grad_norm = 2.0
    best_acc = 0
    best_loss = float('inf')
    count_epoch = 0
    tr_acc_list_e = []
    tr_loss_list_e = []
    tr_loss_c_list_e = []
    val_acc_list_e = []
    val_loss_list_e = []
    val_loss_c_list_e = []
    for epoch in range(num_epochs):
        time_start = time.time()
        tr_acc, tr_loss, tr_loss_c = train_epoch_MDAGCN(model, trGen, loss_func, opt, epoch,
                                                                       max_grad_norm)
        va_acc, va_loss, va_loss_c = val_MDAGCN(model, cvGen, loss_func, epoch, False)
        scheduler.step(va_loss)

        # Save training information
        tr_acc_list_e.append(tr_acc)
        tr_loss_list_e.append(tr_loss)
        tr_loss_c_list_e.append(tr_loss_c)
        val_acc_list_e.append(va_acc)
        val_loss_list_e.append(va_loss)
        val_loss_c_list_e.append(va_loss_c)

        # Save best & Early stopping

        current_metric = va_acc - 0.1 * va_loss

        if current_metric > best_acc - 0.1 * best_loss or va_loss < best_loss:
            if va_acc > best_acc:
                best_acc = va_acc
            if va_loss < best_loss:
                best_loss = va_loss

            torch.save(model.state_dict(), Path['Save'] + 'MDAGCN_Best_' + str(i) + '.pth')
            print(" U ", end='')
            count_epoch = 0
        else:
            count_epoch += 1
            print("   ", end='')
            if count_epoch >= 20:
                print(" ES ")
                break
        time_end = time.time()
        time_cost = time_end - time_start
        print(" Time:%.3f" % (time_cost), f"LR:{opt.param_groups[0]['lr']:.6f}")

    # load the weights of best performance
    model.eval()
    model.load_state_dict(torch.load(Path['Save']+'MDAGCN_Best_'+str(i)+'.pth'))
    
    saveFile = open(Path['Save'] + "Result_MDAGCN.txt", 'a+')
    print('Fold #'+str(i), file=saveFile)
    print('TR_ACC:', tr_acc_list_e, '; TR_loss:', tr_loss_list_e, '; TR_loss_c:', tr_loss_c_list_e,
          '; VA_ACC:', val_acc_list_e, '; VA_loss:', val_loss_list_e, '; VA_loss_c:', val_loss_c_list_e, ';',
          file=saveFile)
    saveFile.close()

    del model, train_targets, val_targets, train_feature, val_feature
    gc.collect()
    
    return tr_loss_list_e, tr_acc_list_e, val_loss_list_e, val_acc_list_e


# # 4. Collect training results

for i in range(fold):
    fit_loss_i, fit_acc_i, fit_val_loss_i, fit_val_acc_i = train_MDAGCN_fold(i)
    
    VariationCurve(fit_acc_i, fit_val_acc_i, 'Acc_MDAGCN_'+str(i), Path['Save'], figsize=(9, 6))
    VariationCurve(fit_loss_i, fit_val_loss_i, 'Loss_MDAGCN_'+str(i), Path['Save'], figsize=(9, 6))
    
    fit_acc.append(fit_acc_i)
    fit_val_loss.append(fit_val_loss_i)
    fit_val_acc.append(fit_val_acc_i)
    fit_loss.append(fit_loss_i)
    

print(128 * '_')
print('End of training MDAGCN.')
print(128 * '#')
