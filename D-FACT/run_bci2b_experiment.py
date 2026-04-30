"""
BCI IV 2b 数据集训练脚本
使用 Model3 模型进行二分类运动想象任务训练
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import time
import random
import argparse
from torch.utils.data import DataLoader
from src.data.bci_iv_2a_loader import load_bci_iv_2a_numpy as load_BCI42_data
from src.data.eeg_dataset import EEGDataset
from src.models.model3 import Model3
from src.training.trainer import baseModel
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
# BCI IV 2b 数据集配置
DATASET_2B_CONFIG = {
    # 数据集路径
    'data_path': 'D:/DeepLearning/EEG-TransNet-main/dataset/raw',
    'label_path': 'D:/DeepLearning/EEG-TransNet-main/dataset/raw',

    # 训练参数
    'batch_size': 128,  # 最佳 batch size
    'lr': 0.001,        # 学习率
    'epochs': 1,        # 训练轮数

    # 模型参数
    'n_channels': 3,   # BCI 2b 电极数量
    'n_classes': 2,     # 二分类任务
    'out_folder': 'output/bci_iv_2b',  # 输出目录
    'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    'random_seed': 42,
    # 敏感度分析最佳参数
    'num_experts': 3,              # 动态卷积专家数量
    'freq_split_ratios': (0.3, 0.3),  # 频率分割比例
    'preferred_device': 'gpu',
    'num_classes': 2,             # 数据增强需要
    'num_segs': 8,                # 数据增强分段数
    'nGPU': 0                     # GPU编号
}

def set_seed(seed):
    """
    设置随机种子以确保实验可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    """
    主函数：运行BCI IV 2b实验
    遍历9个受试者，进行训练和评估
    """
    parser = argparse.ArgumentParser(description="Run BCI IV-2b experiment.")
    parser.add_argument('--epochs', type=int, default=None, help='覆盖训练轮数')
    parser.add_argument('--out-folder', type=str, default=None, help='覆盖输出目录')
    args = parser.parse_args()

    # 1. 设置配置
    config = DATASET_2B_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.out_folder is not None:
        config['out_folder'] = args.out_folder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['random_seed'])

    if not os.path.exists(config['out_folder']):
        os.makedirs(config['out_folder'])

    print(f"Running BCI IV 2b Experiment with Best Configuration")
    print(f"Experts: {config['num_experts']}, Freq Split: {config['freq_split_ratios']}")
    print(f"Epochs: {config['epochs']}")
    print("-" * 50)

    # 2. 遍历受试者 (BCI IV 2b 有9个受试者)
    subjects = range(1, 10)

    avg_acc = []
    avg_kappa = []
    avg_f1 = []

    for sub_id in subjects:
        print(f"\nSubject {sub_id}...")

        # 3. 加载数据
        # BCI IV 2b 文件命名格式: B01T, B01E 等
        train_file = f'B0{sub_id}T'
        test_file = f'B0{sub_id}E'

        try:
            train_data, train_label = load_BCI42_data(config['data_path'], train_file)
            test_data, test_label = load_BCI42_data(config['data_path'], test_file)
        except Exception as e:
            print(f"Error loading subject {sub_id}: {e}")
            print("Please ensure BCI IV 2b data is in .npy format in the specified path.")
            continue

        train_dataset = EEGDataset(train_data, train_label)
        test_dataset = EEGDataset(test_data, test_label)

        # 4. 初始化模型
        # BCI 2b: 3个电极, 2个类别
        net_args = {
            'n_channels': config['n_channels'],
            'n_classes': config['n_classes'],
            'num_experts': config['num_experts'],
            'freq_split_ratios': config['freq_split_ratios']
        }

        net = Model3(**net_args).to(device)

        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=1e-3)

        # 5. 训练
        # 保存到带时间戳的子文件夹
        save_path = os.path.join(config['out_folder'], f'sub{sub_id}', config['timestamp'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_wrapper = baseModel(
            net=net,
            config=config,
            optimizer=optimizer,
            loss_func=loss_func,
            result_savepath=save_path
        )

        acc, kappa, f1 = model_wrapper.train_test(
            train_dataset,
            test_dataset,
            config['epochs'],
            sub_id
        )

        print(f"Subject {sub_id} Result: Acc={acc:.4f}, Kappa={kappa:.4f}, F1={f1:.4f}")

        avg_acc.append(acc)
        avg_kappa.append(kappa)
        avg_f1.append(f1)

    # 6. 输出汇总
    print("\n" + "="*50)
    print("BCI IV 2b Experiment Summary")
    if len(avg_acc) > 0:
        print(f"Average Accuracy: {np.mean(avg_acc):.4f}")
        print(f"Average Kappa:    {np.mean(avg_kappa):.4f}")
        print(f"Average F1-Score: {np.mean(avg_f1):.4f}")
    else:
        print("No subjects processed successfully.")
    print("="*50)

if __name__ == '__main__':
    main()