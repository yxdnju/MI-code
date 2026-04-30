
"""
Model3 训练脚本 - BCI IV 2a 数据集
标准训练流程
"""

import os
import torch
import yaml
import time
import numpy as np

from src.data.bci_iv_2a_loader import load_bci_iv_2a_numpy
from src.data.eeg_dataset import EEGDataset
from src.models.model3 import Model3
from src.training.trainer import baseModel

def dictToYaml(filePath, dictToWrite):
    """将字典写入YAML文件"""
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)

def setRandom(seed):
    """设置随机种子以确保可重复性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def run_experiment(exp_name, num_experts, freq_split_ratios, config):
    """
    运行单个敏感度实验
    
    Args:
        exp_name: 实验名称
        num_experts: 专家数量
        freq_split_ratios: 频率分割比例 (beta, theta) -> mu = 1 - beta - theta
        config: 配置字典
    """
    print(f"\n{'='*50}")
    print(f"Running Experiment: {exp_name}")
    print(f"Parameters: num_experts={num_experts}, freq_split_ratios={freq_split_ratios}")
    print(f"{'='*50}\n")
    
    data_path = os.environ.get('BCI_DATA_PATH', config['data_path'])
    out_folder = config['out_folder']
    # 生成带时间戳的输出文件夹名
    random_folder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime())) + f"_{exp_name}"
    
    # 获取要训练的受试者数量（可通过环境变量覆盖）
    max_subject = int(os.environ.get('SUBJECTS_OVERRIDE', 9))
    subjects = range(1, max_subject + 1)
    
    best_acc_list = []
    best_kappa_list = []
    best_f1_list = []
    
    # 实验结果日志文件
    log_file_path = os.path.join(out_folder, exp_name, "results.txt")
    
    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
        
    # 初始化日志文件
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Parameters: num_experts={num_experts}, freq_split_ratios={freq_split_ratios}\n")
            f.write("="*50 + "\n")
            f.write("Subject | Accuracy | Kappa | F1-Score\n")
    
    for subId in subjects:
        print(f"Subject {subId}...")
        
        # BCI 2a 数据文件命名格式：A01T, A01E 等
        train_datafile = 'A0' + str(subId) + 'T'
        test_datafile = 'A0' + str(subId) + 'E'
        
        # 输出路径：output/数据集/实验名/sub 受试者 ID/时间戳/
        out_path = os.path.join(out_folder, exp_name, 'sub'+str(subId), random_folder)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        # 保存实验配置
        dictToYaml(os.path.join(out_path, 'config.yaml'), config)
        setRandom(config['random_seed'])
        
        # 加载数据
        try:
            train_data, train_labels = load_bci_iv_2a_numpy(data_path, train_datafile)
            test_data, test_labels = load_bci_iv_2a_numpy(data_path, test_datafile)
        except Exception as e:
            print(f"Error loading data for subject {subId}: {e}")
            continue
            
        train_dataset = EEGDataset(train_data, train_labels)
        test_dataset = EEGDataset(test_data, test_labels)
        
        # 初始化带敏感度参数的模型
        net_args = config['network_args'].copy()
        
        if num_experts <= 0:
            raise ValueError("num_experts 必须为正数")

        net_args['num_experts'] = num_experts
        net_args['freq_split_ratios'] = freq_split_ratios
        net = Model3(**net_args)
        
        # 训练设置
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=1e-3)
        
        model = baseModel(net, config, optimizer, loss_func, result_savepath=out_path)
        
        # 训练
        print(f"Subject {subId}: Starting training...")
        best_acc, best_kappa, best_f1 = model.train_test(train_dataset, test_dataset, 1, subId)
        
        best_acc_list.append(best_acc)
        best_kappa_list.append(best_kappa)
        best_f1_list.append(best_f1)
        
        print(f"Subject {subId} Best Acc: {best_acc:.4f}, Kappa: {best_kappa:.4f}, F1: {best_f1:.4f}")
        
        # 记录每个受试者的结果
        with open(log_file_path, 'a') as f:
            f.write(f"Subject {subId} | {best_acc:.4f} | {best_kappa:.4f} | {best_f1:.4f}\n")
        
    # 计算平均结果
    avg_acc = sum(best_acc_list) / len(best_acc_list) if best_acc_list else 0
    avg_kappa = sum(best_kappa_list) / len(best_kappa_list) if best_kappa_list else 0
    avg_f1 = sum(best_f1_list) / len(best_f1_list) if best_f1_list else 0
    
    print(f"\nExperiment {exp_name} Finished.")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average Kappa: {avg_kappa:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")
    
    # 记录平均结果
    with open(log_file_path, 'a') as f:
        f.write("-" * 50 + "\n")
        f.write(f"Average   | {avg_acc:.4f} | {avg_kappa:.4f} | {avg_f1:.4f}\n")
        
    return avg_acc, avg_kappa, avg_f1

def main():
    """主函数：加载配置并运行实验"""
    # 加载基础配置
    base_config_path = os.environ.get(
        'CONFIG_PATH',
        'configs/bci_iv_2a_model3.yaml',
    )
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.full_load(f)
        
    # 允许通过环境变量覆盖数据路径
    if os.environ.get('BCI_DATA_PATH'):
        config['data_path'] = os.environ['BCI_DATA_PATH']

    # 确保网络名称正确用于日志记录
    config['network'] = 'Model3'
    
    # 允许通过环境变量覆盖训练轮数
    if os.environ.get('EPOCHS_OVERRIDE'):
        config['epochs'] = int(os.environ['EPOCHS_OVERRIDE'])
    print(f"Training epochs: {config['epochs']}")
    
    # 频率分割敏感度实验配置
    # Ratio: (Beta, Theta) -> Mu = 1 - Beta - Theta
    experiments = [
        {
            "name": "freq_balanced", # 近似均匀分割
            "num_experts": 3,
            "freq_split": (0.33, 0.33) # Mu=0.34
        }
    ]
    
    results = {}

    # 确保输出目录存在
    os.makedirs(config['out_folder'], exist_ok=True)

    # 全局汇总日志
    global_log_path = os.path.join(config['out_folder'], "summary.txt")
    with open(global_log_path, 'w') as f:
        f.write("Model3 Training Summary\n")
        f.write("="*80 + "\n")
        f.write(f"{'Experiment':<25} | {'Acc':<8} | {'Kappa':<8} | {'F1':<8}\n")
        f.write("-" * 80 + "\n")
    
    print("Starting Model3 Training Experiments...")
    print("NOTE: This will run full training for multiple configurations.")
    
    # 运行所有实验
    for exp in experiments:
        acc, kappa, f1 = run_experiment(
            exp['name'], 
            exp['num_experts'], 
            exp['freq_split'], 
            config
        )
        results[exp['name']] = {'acc': acc, 'kappa': kappa, 'f1': f1}
        
        # 更新全局汇总
        with open(global_log_path, 'a') as f:
            f.write(f"{exp['name']:<25} | {acc:.4f}   | {kappa:.4f}   | {f1:.4f}\n")
        
    # 打印最终结果
    print("\n\nFinal Model3 Training Results:")
    print("="*60)
    print(f"{'Experiment':<25} | {'Acc':<8} | {'Kappa':<8} | {'F1':<8}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['acc']:.4f}   | {metrics['kappa']:.4f}   | {metrics['f1']:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
