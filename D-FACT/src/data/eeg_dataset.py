"""
EEG 数据集类
将 numpy 数据封装为 PyTorch Dataset 格式
"""
import numpy as np
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """
    EEG 数据集类，继承自 PyTorch Dataset

    用于将 EEG 数据和标签封装为可迭代的数据集对象
    """
    def __init__(self, data, label):
        """
        初始化 EEGDataset

        Args:
            data: EEG数据，形状为 (trials, channels, time_points)
            label: 标签数组，形状为 (trials,)
        """
        super().__init__()
        self.labels = label
        self.data = data

    def __getitem__(self, index):
        """
        根据索引获取单个样本

        Args:
            index: 样本索引

        Returns:
            data: 单个EEG样本
            label: 对应的标签
        """
        data = self.data[index]
        label = self.labels[index]

        return data, label

    def __len__(self):
        """
        返回数据集的样本数量

        Returns:
            样本数量
        """
        return len(self.data)