"""
BCI 数据加载模块
支持 BCI IV 2a 和 HGD 数据集的加载
"""
import numpy as np
import random
import os

def load_bci_iv_2a_numpy(dataset_path, data_file):
    """
    加载 BCI IV 2a 数据集的 .npy 文件

    Args:
        dataset_path: 数据集根目录路径
        data_file: 数据文件名（不含后缀），如 'A01T', 'A01E'

    Returns:
        data: EEG数据，形状为 (trials, channels, time_points)
        label: 标签，形状为 (trials,)，值为 0-3
    """
    # 构建数据和标签文件路径
    data_path = os.path.join(dataset_path, data_file + '_data.npy')
    label_path = os.path.join(dataset_path, data_file + '_label.npy')

    # 加载数据
    data = np.load(data_path)
    # 标签减1：将1-4转换为0-3
    label = np.load(label_path).squeeze() - 1

    print(data_file, 'load success')

    # 打乱数据顺序
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def load_HGD_data(dataset_path, data_file, label_file):
    """
    加载 HGD 数据集的 .npy 文件

    Args:
        dataset_path: 数据集根目录路径
        data_file: 数据文件名（含后缀）
        label_file: 标签文件名（含后缀）

    Returns:
        data: EEG数据
        label: 标签
    """
    data = []
    label = []
    data_path = os.path.join(dataset_path, data_file)
    label_path = os.path.join(dataset_path, label_file)

    data = np.load(data_path)
    label = np.load(label_path).squeeze()

    print(data_file, 'load success')

    # 打乱数据顺序
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def shuffle_data(data, label):
    """
    随机打乱数据和标签的对应关系

    Args:
        data: 输入数据
        label: 输入标签

    Returns:
        shuffle_data: 打乱后的数据
        shuffle_label: 打乱后的标签
    """
    index = [i for i in range(len(data))]
    random.shuffle(index)
    shuffle_data = data[index]
    shuffle_label = label[index]
    return shuffle_data, shuffle_label