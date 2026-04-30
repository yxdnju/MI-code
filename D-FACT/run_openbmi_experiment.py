"""
OpenBMI 数据集训练脚本
使用 Model3 模型进行二分类运动想象任务训练

数据来源:
- Session 1: 训练数据
- Session 2: 测试数据
- 共 54 个受试者
"""
import os
import sys
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from src.data.eeg_dataset import EEGDataset as eegDataset
from src.models.model3 import Model3
from src.training.trainer import baseModel
from openbmi_paths import default_openbmi_root, resolve_openbmi_root, subject_files

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
OPENBMI_CONFIG = {
    'n_classes': 2,
    'n_channels': 62,  # OpenBMI 有62个电极
    'data_path': str(default_openbmi_root()),
    'out_folder': 'output/openbmi',
    'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    'epochs': 1,
    'batch_size': 32,
    'lr': 0.001,
    'random_seed': 42,
    # 敏感度分析最佳参数
    'num_experts': 3,
    'freq_split_ratios': (0.3, 0.3),
    'preferred_device': 'gpu',
    'num_classes': 2,
    'num_segs': 8,
    'nGPU': 0,
    'sampling_rate': 250,   # 目标采样率
    'original_fs': 1000     # 原始采样率
}

def set_seed(seed):
    """
    设置随机种子以确保实验可复现
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def load_openbmi_data(config, sub_id):
    """
    加载指定受试者的OpenBMI数据

    读取Session 1和Session 2的数据
    Session 1作为训练集
    Session 2作为测试集

    Args:
        config: 配置字典
        sub_id: 受试者ID

    Returns:
        X_train: 训练数据
        y_train: 训练标签
        X_test: 测试数据
        y_test: 测试标签
    """
    print(f"Loading data for Subject {sub_id}...")

    # 获取文件路径
    base_path = resolve_openbmi_root(config.get('data_path'))
    sess1_path, sess2_path = subject_files(base_path, sub_id)
    sess1_path = os.fspath(sess1_path)
    sess2_path = os.fspath(sess2_path)

    if not os.path.exists(sess1_path) or not os.path.exists(sess2_path):
        raise FileNotFoundError(f"Data files for subject {sub_id} not found at {sess1_path} or {sess2_path}")

    # 加载Session 1 (训练)
    X_train, y_train = _load_session(sess1_path, config)

    # 加载Session 2 (测试)
    X_test, y_test = _load_session(sess2_path, config)

    print(f"Subject {sub_id} loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def _load_session(file_path, config):
    """
    加载单个session的.mat文件

    Args:
        file_path: .mat文件路径
        config: 配置字典

    Returns:
        X: EEG数据，形状 (Trials, Channels, Time)
        y: 标签，形状 (Trials,)
    """
    try:
        mat = sio.loadmat(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise e

    # OpenBMI结构: 每个session文件包含EEG_MI_train和EEG_MI_test

    # 提取训练部分
    train_struct = mat['EEG_MI_train'][0, 0]
    X1 = train_struct['smt']  # (Time, Trials, Channels)
    y1 = train_struct['y_dec'][0]  # (Trials,)

    # 提取测试部分
    test_struct = mat['EEG_MI_test'][0, 0]
    X2 = test_struct['smt']
    y2 = test_struct['y_dec'][0]

    # 沿Trials轴拼接
    X = np.concatenate((X1, X2), axis=1)
    y = np.concatenate((y1, y2))

    # 转置为 (Trials, Channels, Time)
    X = np.transpose(X, (1, 2, 0))

    # 降采样 (从1000Hz降到250Hz)
    orig_fs = config['original_fs']
    target_fs = config['sampling_rate']
    if orig_fs != target_fs:
        factor = int(orig_fs / target_fs)
        X = signal.decimate(X, factor, axis=-1)

    # 标签调整: (1, 2) -> (0, 1)
    y = y - 1

    # 标准化 (每个trial独立标准化)
    mean = np.mean(X, axis=2, keepdims=True)
    std = np.std(X, axis=2, keepdims=True)
    X = (X - mean) / (std + 1e-6)

    return X.astype(np.float32), y.astype(np.longlong)

def main():
    """
    主函数：运行OpenBMI实验
    遍历54个受试者，进行训练和评估
    支持跳过已完成和失败的受试者
    """
    parser = argparse.ArgumentParser(description="Run OpenBMI experiment.")
    parser.add_argument('--epochs', type=int, default=None, help='覆盖训练轮数')
    parser.add_argument('--out-folder', type=str, default=None, help='覆盖输出目录')
    args = parser.parse_args()

    # 1. 设置配置
    config = OPENBMI_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.out_folder is not None:
        config['out_folder'] = args.out_folder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['random_seed'])

    if not os.path.exists(config['out_folder']):
        os.makedirs(config['out_folder'])

    print(f"Running OpenBMI Experiment")
    print(f"Channels: {config['n_channels']}, Classes: {config['n_classes']}")
    print(f"Epochs: {config['epochs']}")
    print("-" * 50)

    # 2. 遍历受试者 (OpenBMI有54个受试者)
    subjects = list(range(1, 55))

    # 记录指标
    avg_acc = []
    avg_kappa = []
    avg_f1 = []

    # 记录结果
    completed_subjects = []
    skipped_subjects = []
    failed_subjects = []

    for sub_id in subjects:
        # 检查是否已完成
        sub_dir = os.path.join(config['out_folder'], f'sub{sub_id}')
        is_completed = False
        if os.path.exists(sub_dir):
            # 检查直接文件
            for f in os.listdir(sub_dir):
                if f.startswith('log_result') and f.endswith('.txt'):
                    try:
                        with open(os.path.join(sub_dir, f), 'r') as log_f:
                            content = log_f.read()
                            if "The best accuracy is:" in content:
                                is_completed = True
                                print(f"Skipping Subject {sub_id} (Found completed run in {f})")
                                break
                    except:
                        pass

            if not is_completed:
                # 检查子目录(带时间戳的运行)
                for run_dir in os.listdir(sub_dir):
                    run_full_path = os.path.join(sub_dir, run_dir)
                    if not os.path.isdir(run_full_path): continue

                    for f in os.listdir(run_full_path):
                        if f.startswith('log_result') and f.endswith('.txt'):
                            try:
                                with open(os.path.join(run_full_path, f), 'r') as log_f:
                                    content = log_f.read()
                                    if "The best accuracy is:" in content:
                                        is_completed = True
                                        print(f"Skipping Subject {sub_id} (Found completed run in {run_dir}/{f})")
                                        break
                            except:
                                pass
                    if is_completed: break

        if is_completed:
            skipped_subjects.append(sub_id)
            continue

        print(f"\nProcessing Subject {sub_id}...")

        # 3. 加载数据
        try:
            train_data, train_label, test_data, test_label = load_openbmi_data(config, sub_id)
        except Exception as e:
            print(f"Skipping Subject {sub_id}: {e}")
            failed_subjects.append({'subject_id': sub_id, 'error': str(e)})
            continue

        train_dataset = eegDataset(train_data, train_label)
        test_dataset = eegDataset(test_data, test_label)

        # 4. 初始化模型
        net_args = {
            'n_channels': config['n_channels'],
            'n_classes': config['n_classes'],
            'num_experts': config['num_experts'],
            'freq_split_ratios': config['freq_split_ratios']
        }

        net = Model3(**net_args).to(device)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
        loss_func = nn.CrossEntropyLoss()

        # 5. 训练
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

        # 训练并获取结果
        acc, kappa, f1 = model_wrapper.train_test(train_dataset, test_dataset, dataset=3, subId=sub_id)

        print(f"Subject {sub_id} Results - Acc: {acc:.4f}, Kappa: {kappa:.4f}, F1: {f1:.4f}")

        avg_acc.append(acc)
        avg_kappa.append(kappa)
        avg_f1.append(f1)

        # 记录完成的受试者
        completed_subjects.append({
            'subject_id': sub_id,
            'accuracy': acc,
            'kappa': kappa,
            'f1': f1
        })

        # 清理内存
        del net, model_wrapper, train_dataset, test_dataset, train_data, test_data
        torch.cuda.empty_cache()

    # 6. 输出汇总
    print("\n" + "="*50)
    print("OpenBMI Experiment Summary")
    if len(avg_acc) > 0:
        print(f"Average Accuracy: {np.mean(avg_acc):.4f}")
        print(f"Average Kappa:    {np.mean(avg_kappa):.4f}")
        print(f"Average F1-Score: {np.mean(avg_f1):.4f}")
    else:
        print("No subjects processed successfully.")

    # 输出详细结果
    print("\n" + "="*50)
    print("Detailed Results")
    print("="*50)

    print(f"\n✅ Completed subjects ({len(completed_subjects)}):")
    if completed_subjects:
        for item in completed_subjects:
            print(f"   Sub{item['subject_id']}: Acc={item['accuracy']:.4f}, Kappa={item['kappa']:.4f}, F1={item['f1']:.4f}")
    else:
        print("   (None)")

    print(f"\n⏭️  Skipped subjects (already completed) ({len(skipped_subjects)}):")
    if skipped_subjects:
        print(f"   {skipped_subjects}")
    else:
        print("   (None)")

    print(f"\n❌ Failed subjects ({len(failed_subjects)}):")
    if failed_subjects:
        for item in failed_subjects:
            print(f"   Sub{item['subject_id']}: {item['error']}")
    else:
        print("   (None)")

    print("="*50)

if __name__ == '__main__':
    main()