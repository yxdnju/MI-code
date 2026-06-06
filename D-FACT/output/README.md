# Output 文件夹说明

本目录包含所有实验的输出结果，包括模型权重、训练日志、配置文件、混淆矩阵和统计分析结果。

---

## 目录结构

```
output/
── HGD/                          # HGD 数据集实验结果
├── bci_iv_2a/                    # BCI IV-2a 数据集实验结果
├── bci_iv_2b/                    # BCI IV-2b 数据集实验结果
├── confusion_matrices/           # 混淆矩阵可视化结果
└── tuning_temp/                  # 临时调参实验结果
```

---

## 详细说明

### 1. HGD/ - HGD 数据集实验

HGD（High Gamma Dataset）数据集包含 14 名受试者的运动想象 EEG 数据，4 分类任务。

#### Model3_Yuan/
原始 Model3 模型在 HGD 数据集上的训练结果。
- **结构**: `sub{1-14}/{timestamp}/`
- **内容**: `config.yaml`（配置文件）、`log_result_*.txt`（训练日志）、`model.pth`（模型权重）
- **受试者**: sub1 - sub14

#### Model3_withoutDynamic/
移除动态卷积模块的消融实验结果。
- **受试者**: sub1 - sub14

#### Model3_withoutFreq/
移除频带注意力模块的消融实验结果。
- **受试者**: sub1 - sub14

#### Model3_withoutLastStep/
移除最后一个时间步的消融实验结果。
- **受试者**: sub1 - sub14

#### hgd_model3_subject_results.md
HGD 数据集各受试者的实验结果汇总文档。

---

### 2. bci_iv_2a/ - BCI IV-2a 数据集实验

BCI Competition IV 2a 数据集，包含 9 名受试者，4 分类运动想象任务。

#### loso_experiment/
Leave-One-Subject-Out 交叉验证实验结果。
- **文件**: `results.txt`

#### sensitivity_experts_{1,2,4,5}/
不同专家数量的敏感性分析实验。

#### sensitivity_freq_{balanced,default,high_focus,low_focus}/
频带注意力不同配置的敏感性分析。

#### sensitivity_no_dynamic_conv/
移除动态卷积的敏感性分析。

#### 最佳日志/
最优实验配置的日志记录。

#### sensitivity_summary_freq.txt
频带敏感性分析汇总。

---

### 3. bci_iv_2b/ - BCI IV-2b 数据集实验

BCI Competition IV 2b 数据集，包含 9 名受试者，2 分类运动想象任务。

#### Model3/
原始 Model3 模型训练结果。
- **结构**: `sub{1-9}/{timestamp}/`
- **内容**: `config.yaml`、`log_result.txt`

#### Model3_withoutDynamic/
移除动态卷积的消融实验。
- **受试者**: sub1 - sub9

#### Model3_withoutFreq/
移除频带注意力的消融实验。
- **受试者**: sub1 - sub9

#### Model3_withoutTCN/
移除 TCN 模块的消融实验。
- **受试者**: sub1 - sub9

---

### 4. confusion_matrices/ - 混淆矩阵

各数据集、模型、受试者的混淆矩阵可视化结果。每个混淆矩阵包含 `.png` 热力图和 `.txt` 详细指标文件。

#### bci_iv_2a/
BCI IV-2a 数据集的混淆矩阵。
- **模型**: Model3、Model3_withoutDynamic、Model3_withoutFreq、Model3_withoutTCN
- **受试者**: sub01 - sub09
- **命名格式**: `bci_iv_2a_sub{XX}_Model3.png` 和 `bci_iv_2a_sub{XX}_Model3.txt`

#### bci_iv_2b/
BCI IV-2b 数据集的混淆矩阵。
- **模型**: Model3_Src（原始模型）
- **受试者**: sub01 - sub09
- **命名格式**: `bci_iv_2b_sub{XX}_Model3_Src.png` 和 `bci_iv_2b_sub{XX}_Model3_Src.txt`

#### hgd/
HGD 数据集的混淆矩阵。
- **模型**: Model3、HGD_Model3_withoutDynamic、HGD_Model3_withoutFreq、HGD_Model3_withoutLastStep
- **受试者**: sub01 - sub14
- **命名格式**: `hgd_sub{XX}_Model3.png` 和 `hgd_sub{XX}_Model3.txt`

#### openbmi/
OpenBMI 数据集的混淆矩阵。
- **模型**: Model3_Src（原始模型）
- **受试者**: sub01 - sub54
- **命名格式**: `openbmi_sub{XX}_Model3_Src.png` 和 `openbmi_sub{XX}_Model3_Src.txt`

---

### 5. tuning_temp/ - 临时调参实验

临时参数调优实验的中间结果，包含 OpenBMI 数据集各受试者的多次调参尝试。
- **结构**: `output/openbmi/sub{1-54}/{timestamp}/`
- **内容**: `config.yaml`（配置文件）、`log_result_*.txt`（训练日志）、`model.pth`（模型权重）
- **用途**: 用于超参数搜索和模型调优的中间产物，包含多个时间戳的实验记录

---

## 文件说明

### config.yaml
每次实验的配置文件，包含：
- 超参数（学习率、batch size、epochs 等）
- 模型配置（通道数、类别数等）
- 设备配置（GPU/CPU）

### log_result_*.txt
训练日志文件，包含：
- 每个 epoch 的训练/测试损失和准确率
- Kappa 系数、F1 分数等指标
- 最佳准确率记录
- 训练时间和 ETA

### model.pth
训练保存的模型权重文件

### results.txt / summary.txt
实验结果汇总文件，包含各受试者的平均性能指标。

### confusion_matrices/*.png
混淆矩阵热力图可视化，展示模型在各分类上的预测分布。

### confusion_matrices/*.txt
混淆矩阵详细指标文件，包含：
- TP（True Positive）：真正例
- TN（True Negative）：真负例
- FP（False Positive）：假正例
- FN（False Negative）：假负例
- Precision、Recall、F1-Score 等指标

---


