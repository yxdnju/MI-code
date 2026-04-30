"""
OpenBMI 数据路径解析模块
根据受试者ID返回对应的session文件路径
"""
import os
from pathlib import Path

def default_openbmi_root():
    """返回OpenBMI数据集的根目录"""
    return Path('D:/DeepLearning/model3_exp/data/openbmi_mi')

def resolve_openbmi_root(base_path):
    """解析OpenBMI数据集根路径"""
    return Path(base_path)

def subject_files(base_path, sub_id):
    """
    根据受试者ID返回session1和session2的文件路径

    Args:
        base_path: OpenBMI数据集根目录
        sub_id: 受试者ID (1-54)

    Returns:
        (session1文件路径, session2文件路径)
    """
    base_path = Path(base_path)
    folder_str = f"{sub_id}"       # 文件夹名称: s1, s2, ..., s54
    file_str = f"{sub_id:02d}"      # 文件名中的ID: 01, 02, ..., 54
    sess1_filename = f"session1/s{folder_str}/sess01_subj{file_str}_EEG_MI.mat"
    sess2_filename = f"session2/s{folder_str}/sess02_subj{file_str}_EEG_MI.mat"
    return base_path / sess1_filename, base_path / sess2_filename