"""
训练器模块
提供模型训练、测试和评估功能
"""
import collections

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import os
import copy
import itertools
from mpl_toolkits.axes_grid1 import host_subplot
from datetime import datetime
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau


def save_confusion_matrix(cm, dataset, sub_id):
    """
    保存混淆矩阵为热力图

    Args:
        cm: 混淆矩阵
        dataset: 数据集标识 (1=BCI 2a四分类, 其他=二分类)
        sub_id: 受试者ID
    """
    filename = f'accuracy_matrix_sub_{sub_id}.png'
    column_sums = cm.sum(axis=0)

    # 计算准确率矩阵
    accuracy_matrix = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if column_sums[j] > 0:
                accuracy_matrix[i, j] = cm[i, j] / column_sums[j]
    plt.figure(figsize=(8, 6))
    class_names = ['Left', 'Right', 'Feet', 'Tongue']
    class_names1 = ['Left', 'Right']
    if dataset == 1:
        sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                    xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                    xticklabels=class_names1, yticklabels=class_names1)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if dataset == 1:
        out_dir = 'output/bci_iv_2a/confusion_matrix'
    else:
        out_dir = 'output/bci_iv_2b/confusion_matrix'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, filename))


class baseModel():
    """
    基础模型训练类
    提供完整的训练、验证和模型保存流程
    """
    def __init__(self, net, config, optimizer, loss_func, scheduler=None, result_savepath=None):
        """
        初始化训练器

        Args:
            net: PyTorch模型
            config: 配置字典
            optimizer: 优化器
            loss_func: 损失函数
            scheduler: 学习率调度器(可选)
            result_savepath: 结果保存路径(可选)
        """
        self.batchsize = config['batch_size']
        self.epochs = config['epochs']
        self.preferred_device = config['preferred_device']
        self.num_workers = int(config.get('num_workers', 0))
        self.pin_memory = bool(config.get('pin_memory', True))

        self.num_classes = config['num_classes']
        self.num_segs = config['num_segs']

        self.device = None
        self.set_device(config['nGPU'])
        self.net = net.to(self.device)

        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = scheduler

        self.result_savepath = result_savepath
        self.log_write = None
        if self.result_savepath is not None:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = f'log_result_{current_time}.txt'
            self.log_write = open(os.path.join(self.result_savepath, log_filename), 'w')

    def set_device(self, nGPU):
        """设置计算设备(CPU/GPU)"""
        if self.preferred_device == 'gpu':
            self.device = torch.device('cuda:' + str(nGPU) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print("Code will be running on device ", self.device)

    def data_augmentation(self, data, label):
        """
        数据增强：对EEG信号进行分段混合增强

        将每个样本分成num_segs段，然后随机混合不同样本的段，
        以增加训练数据的多样性

        Args:
            data: EEG数据，形状 (N, C, T)
            label: 标签，形状 (N,)

        Returns:
            aug_data: 增强后的数据
            aug_label: 增强后的标签
        """
        aug_data = []
        aug_label = []

        N, C, T = data.shape
        seg_size = T // self.num_segs
        aug_data_size = N // self.num_classes

        for cls in range(self.num_classes):
            cls_idx = (label == cls).nonzero(as_tuple=True)[0]
            cls_data = data[cls_idx]
            data_size = cls_data.shape[0]
            if data_size == 0 or data_size == 1:
                continue

            temp_aug_data = torch.zeros((aug_data_size, C, T), device=data.device, dtype=data.dtype)

            for i in range(aug_data_size):
                rand_idx = torch.randint(0, data_size, (self.num_segs,), device=data.device)
                for j in range(self.num_segs):
                    temp_aug_data[i, :, j * seg_size:(j + 1) * seg_size] = cls_data[rand_idx[j], :,
                                                                           j * seg_size:(j + 1) * seg_size]
            aug_data.append(temp_aug_data)
            aug_label.extend([cls] * aug_data_size)

        if not aug_data:
             return torch.empty((0, C, T), device=data.device), torch.empty((0,), device=data.device, dtype=label.dtype)

        aug_data = torch.cat(aug_data, dim=0)
        aug_label = torch.tensor(aug_label, device=data.device, dtype=label.dtype)

        aug_shuffle = torch.randperm(len(aug_data), device=data.device)
        aug_data = aug_data[aug_shuffle]
        aug_label = aug_label[aug_shuffle]

        return aug_data, aug_label

    def train_test(self, train_dataset, test_dataset, dataset, subId):
        """
        训练并测试模型

        Args:
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            dataset: 数据集标识
            subId: 受试者ID

        Returns:
            best_acc: 最佳准确率
            best_kappa: 最佳Cohen's Kappa分数
            best_test_f_score: 最佳F1分数
        """
        global test_f_score, best_kappa
        use_persistent_workers = self.num_workers > 0
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=use_persistent_workers
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batchsize,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=use_persistent_workers
        )
        best_cm = None
        best_acc = 0
        avg_acc = 0
        best_epoch = 0
        best_test_f_score = 0
        best_model = None
        self.best_kappa = 0
        self.best_f_score = 0
        self.best_cm = None
        # 使用余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-5
        )
        self._eta_epoch_time_sum = 0.0
        for epoch in range(self.epochs):
            start_time = time.time()
            self.net.train()
            train_loss = 0
            train_predicted = []
            train_actual = []
            aug_data = 0
            aug_label = 0
            # 每30个epoch重置优化器状态
            if epoch % 30 == 0:
                self.optimizer.state = collections.defaultdict(dict)
            with torch.enable_grad():
                for train_data, train_label in train_dataloader:
                    train_data = train_data.type(torch.FloatTensor).to(self.device)
                    train_label = train_label.type(torch.LongTensor).to(self.device)

                    train_output = self.net(train_data)

                    running_train_loss = self.loss_func(train_output, train_label)
                    self.optimizer.zero_grad()
                    running_train_loss.backward()
                    self.optimizer.step()

                    train_loss += running_train_loss.item()

                    train_predicted.extend(torch.max(train_output, 1)[1].cpu().tolist())
                    train_actual.extend(train_label.cpu().tolist())

            train_loss /= len(train_dataloader)

            if self.scheduler is not None:
                self.scheduler.step(train_loss)

            # 测试阶段
            self.net.eval()
            test_loss = 0
            test_predicted = []
            test_actual = []
            with torch.no_grad():
                for test_data, test_label in test_dataloader:
                    test_data = test_data.type(torch.FloatTensor).to(self.device)
                    test_label = test_label.type(torch.LongTensor).to(self.device)

                    test_output = self.net(test_data)

                    running_test_loss = self.loss_func(test_output, test_label)

                    test_predicted.extend(torch.max(test_output, 1)[1].cpu().tolist())
                    test_actual.extend(test_label.cpu().tolist())
                    test_loss += running_test_loss

            test_loss /= len(test_dataloader)

            # 计算评估指标
            train_acc = accuracy_score(train_actual, train_predicted)
            test_acc = accuracy_score(test_actual, test_predicted)
            test_kappa = cohen_kappa_score(test_actual, test_predicted)
            test_f_score = f1_score(test_actual, test_predicted, average='weighted')

            avg_acc += test_acc

            # 记录最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                best_kappa = test_kappa
                best_test_f_score = test_f_score
                y_true = test_actual
                y_pred = test_predicted
                best_cm = confusion_matrix(y_true, y_pred)
                self.best_cm = best_cm
                self.best_kappa = best_kappa
                self.best_f_score = best_test_f_score
                best_model = copy.deepcopy(self.net.state_dict())

            # 计算剩余时间
            end_time = time.time()
            train_time = end_time - start_time
            elapsed_epochs = epoch + 1
            self._eta_epoch_time_sum += train_time
            avg_epoch_time = self._eta_epoch_time_sum / elapsed_epochs
            remain_epochs = self.epochs - elapsed_epochs
            eta_seconds = avg_epoch_time * remain_epochs
            eta_min = eta_seconds / 60.0

            if self.log_write:
                self.log_write.write(
                    f'Epoch [{epoch + 1}] | Train Loss: {train_loss:.6f}  Train Accuracy: {train_acc:.6f} | Test Loss: {test_loss:.6f} Test Accuracy: {test_acc:.6f} Test Kappa: {test_kappa:.6f}  \n')
            print(
                'Epoch [%d] | Train Loss: %.6f  Train Accuracy: %.6f | Test Loss: %.6f  Test Accuracy: %.6f | Best Acc: %.6f lr: %.6f | train_time: %.6f | ETA: %.2f min | f_score: %.6f '
                % (
                    epoch + 1, train_loss, train_acc, test_loss, test_acc, best_acc,
                    self.optimizer.param_groups[0]['lr'],
                    train_time, eta_min, test_f_score))

        avg_acc /= self.epochs
        print('The average accuracy is: ', avg_acc)
        print('The best accuracy is: ', best_acc)
        print('The best epoch is: ', best_epoch)
        print('The best f_score is: ', best_test_f_score)
        if self.log_write:
            self.log_write.write(f'The average accuracy is: {avg_acc:.6f}\n')
            self.log_write.write(f'The best accuracy is: {best_acc:.6f}\n')
            self.log_write.write(f'The best epoch is: {best_epoch:.6f}\n')
            self.log_write.write(f'The best kappa is: {best_kappa:.6f}\n')
            self.log_write.write(f'The best f_score is: {best_test_f_score:.6f}\n')
            self.log_write.close()

        # 保存最佳模型
        torch.save(best_model, os.path.join(self.result_savepath, 'model.pth'))
        return best_acc, best_kappa, best_test_f_score