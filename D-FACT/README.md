# Model3 BCI 多数据集训练框架

本项目是基于 Model3 模型的多数据集训练框架，支持 BCI IV 2a、BCI IV 2b 和 OpenBMI 三个数据集的训练。

**注意：本项目不包含数据集，数据需要自行下载。**

## 项目目录结构

```
model3_sensitivity_bci2a/
├── train_model3.py                  # BCI IV 2a 训练入口
├── run_bci2b_experiment.py          # BCI IV 2b 训练入口
├── run_openbmi_experiment.py        # OpenBMI 训练入口
├── openbmi_paths.py                 # OpenBMI 数据路径解析
│
├── configs/                         # 配置文件目录
│   ├── bci_iv_2a_model3.yaml        # BCI 2a 配置文件
│   └── bciiv2b_transnet.yaml        # BCI 2b 配置文件
│
├── src/                             # 核心源代码
│   ├── data/                        # 数据处理
│   │   ├── bci_iv_2a_loader.py     # BCI 数据加载器
│   │   └── eeg_dataset.py          # EEG 数据集类
│   ├── models/                      # 模型定义
│   │   └── model3.py                # Model3 模型
│   └── training/                    # 训练相关
│       └── trainer.py               # 训练器基类
│
├── checkpoints/                     # 模型 checkpoint 保存目录
│   └── best_model3/                 # 各受试者最佳模型
│
├── data/                            # 数据目录（本地挂载点，不提交到 git）
│   └── bci_iv_2a_numpy/            # BCI 2a 数据存放位置
│
├── output/                          # 训练输出目录（自动创建）
│   ├── bci_iv_2a/                  # BCI 2a 训练结果
│   ├── bci_iv_2b/                  # BCI 2b 训练结果
│   └── openbmi/                    # OpenBMI 训练结果
│
├── requirements.txt                 # Python 依赖
└── README.md                        # 本文档
```

## 文件说明

### 训练脚本

| 文件 | 说明 |
|------|------|
| `train_model3.py` | BCI IV 2a 训练入口 |
| `run_bci2b_experiment.py` | BCI IV 2b 数据集训练脚本 |
| `run_openbmi_experiment.py` | OpenBMI 数据集训练脚本 |

### 配置文件 (configs/)

| 文件 | 说明 |
|------|------|
| `bci_iv_2a_model3.yaml` | BCI 2a 模型和数据配置 |
| `bciiv2b_transnet.yaml` | BCI 2b 模型和数据配置 |

### 核心源代码 (src/)

| 文件 | 说明 |
|------|------|
| `bci_iv_2a_loader.py` | BCI 系列数据集的加载器，包含 `load_bci_iv_2a_numpy` 和 `load_HGD_data` 函数 |
| `eeg_dataset.py` | PyTorch Dataset 实现，用于 EEG 数据封装 |
| `model3.py` | Model3 模型定义，包含 DynamicTemporalConv 等组件 |
| `trainer.py` | 训练器基类 `baseModel`，包含训练、测试、评估逻辑 |

### 其他文件

| 文件 | 说明 |
|------|------|
| `openbmi_paths.py` | OpenBMI 数据路径解析，根据受试者ID返回对应session文件路径 |

## 数据集说明

本项目支持以下三个数据集，**数据需要自行下载**：

### 1. BCI IV 2a

- **下载地址**：BNCIHorizon2020 或联系数据集提供方
- **数据格式**：`.npy` 格式
- **数据命名**：`A01T_data.npy`, `A01T_label.npy`, `A01E_data.npy`, `A01E_label.npy` 等
- **路径配置**：修改 `configs/bci_iv_2a_model3.yaml` 中的 `data_path`

### 2. BCI IV 2b

- **下载地址**：BNCIHorizon2020 或联系数据集提供方
- **数据格式**：`.npy` 格式
- **数据命名**：`B01T_data.npy`, `B01T_label.npy`, `B01E_data.npy`, `B01E_label.npy` 等
- **路径配置**：修改 `run_bci2b_experiment.py` 中的 `DATASET_2B_CONFIG['data_path']`

### 3. OpenBMI

- **下载地址**：OpenBMI 公开数据集
- **数据格式**：`.mat` 格式（MATLAB格式）
- **目录结构**：
  ```
  openbmi_mi/
  ├── session1/
  │   ├── s1/sess01_subj01_EEG_MI.mat
  │   ├── s2/sess01_subj02_EEG_MI.mat
  │   └── ...
  └── session2/
      ├── s1/sess02_subj01_EEG_MI.mat
      ├── s2/sess02_subj02_EEG_MI.mat
      └── ...
  ```
- **路径配置**：修改 `openbmi_paths.py` 中的 `default_openbmi_root()` 返回值

## 环境配置

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 依赖包说明 (requirements.txt)

```
torch                    # PyTorch深度学习框架
numpy                    # 数值计算
scikit-learn             # 机器学习工具（准确率、kappa、F1等指标）
PyYAML                   # YAML配置文件解析
einops                   # 张量重排操作
scipy                    # 科学计算（OpenBMI数据降采样）
matplotlib               # 可视化（混淆矩阵）
seaborn                  # 统计可视化
```

### 3. 数据路径配置

克隆项目后，需要根据实际数据存放位置修改配置文件中的数据路径：

#### BCI 2a
修改 `configs/bci_iv_2a_model3.yaml`：
```yaml
data_path: D:/你的路径/bci_iv_2a1
```

#### BCI 2b
修改 `run_bci2b_experiment.py` 中的 `DATASET_2B_CONFIG`：
```python
'data_path': 'D:/你的路径/raw',
```

#### OpenBMI
修改 `openbmi_paths.py` 中的 `default_openbmi_root()`：
```python
def default_openbmi_root():
    return Path('D:/你的路径/openbmi_mi')
```

## 快速开始

### BCI IV 2a 训练

```powershell
python train_model3.py
```

可选环境变量：
```powershell
$env:BCI_DATA_PATH="D:\你的路径\bci_iv_2a1"
$env:EPOCHS_OVERRIDE="1"       # 覆盖配置文件中的 epoch 数
$env:SUBJECTS_OVERRIDE="1"     # 覆盖要训练的受试者数量
python train_model3.py
```

### BCI IV 2b 训练

```powershell
python run_bci2b_experiment.py --epochs 1
```

### OpenBMI 训练

```powershell
python run_openbmi_experiment.py --epochs 1
```

## 训练输出

训练完成后，结果会保存在 `output/` 目录下：

```
output/
├── bci_iv_2a/
│   ├── sensitivity_freq_balanced/      # 实验名称
│   │   ├── sub1/                       # 受试者1
│   │   │   └── 2026-04-30--15-30_freq_balanced/
│   │   │       ├── config.yaml        # 实验配置
│   │   │       ├── log_result_*.txt   # 训练日志
│   │   │       └── model.pth          # 最佳模型
│   │   └── ...
│   └── results.txt                     # 所有受试者汇总结果
│
├── bci_iv_2b/                          # 同上结构
│   └── ...
│
└── openbmi/                            # 同上结构
    └── ...
```

## 模型配置参数

主要配置参数（见各配置文件）：

| 参数 | 说明 | BCI 2a | BCI 2b | OpenBMI |
|------|------|--------|--------|---------|
| `n_channels` | EEG通道数 | 22 | 3 | 62 |
| `n_classes` | 分类类别数 | 4 | 2 | 2 |
| `num_experts` | 专家数量 | 3 | 3 | 3 |
| `freq_split_ratios` | 频率分割比例 | (0.33, 0.33) | (0.3, 0.3) | (0.3, 0.3) |
| `batch_size` | 批次大小 | 128 | 128 | 32 |
| `epochs` | 训练轮数 | 900 | 1500 | 1000 |
| `lr` | 学习率 | 0.001 | 0.001 | 0.001 |

## 注意事项

1. **数据自行下载**：本项目不包含任何数据集文件，需要自行从官方渠道下载
2. **路径格式**：Windows系统建议使用正斜杠 `/` 或双反斜杠 `\\`
3. **GPU支持**：如需使用GPU，确保安装支持CUDA的PyTorch版本
4. **输出目录**：训练过程中会自动创建 `output/` 目录及其子目录