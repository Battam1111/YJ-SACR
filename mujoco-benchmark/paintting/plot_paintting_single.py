import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def read_csv_file(file_path: str) -> pd.DataFrame:
    """读取CSV文件并返回DataFrame。"""
    return pd.read_csv(file_path)

def plot_data(df: pd.DataFrame, columns: list, xlabel: str, ylabel: str, title: str, output_file: str, window_size: int = 100):
    """绘制数据图表并保存到文件。"""
    plt.figure()
    for column in columns:
        plt.plot(df['epoch'], df[column], label=f"{column} (Original)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    # 绘制平滑数据图
    plt.figure()
    for column in columns:
        smooth_mean = df[column].rolling(window=window_size).mean()
        plt.plot(df['epoch'], smooth_mean, label=f"{column} (Smoothed)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Smoothed {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_smoothed.png'))
    plt.close()

def extract_info_from_path(file_path: str) -> Tuple[str, str, str, str]:
    """从文件路径中提取任务名称、AF参数、数据类型和种子。"""
    parts = file_path.split('/')
    task_name = parts[-4]
    af_param = parts[-3]
    data_type = parts[-2]
    seed = parts[-1].split('_')[0]
    return task_name, af_param, data_type, seed

def plot_single_experiment(csv_file: str, output_dir: str, window_size: int = 100):
    """读取单个实验的CSV文件并绘制图表。"""
    os.makedirs(output_dir, exist_ok=True)
    df = read_csv_file(csv_file)

    task_name, af_param, data_type, seed = extract_info_from_path(csv_file)
    columns = df.columns[1:]  # 去掉epoch列，保留其他数据列

    plot_data(df, columns, 'Epoch', 'Value', f'{data_type} vs Epoch', os.path.join(output_dir, f'{task_name}_{af_param}_{data_type}_{seed}.png'), window_size)

# 使用示例
single_csv_file = 'mujoco-benchmark/data/Ant-v4/AF0.2/TestLen/Seed0_20240725-133642.csv'
output_dir_single = 'mujoco-benchmark/data/Ant-v4/AF0.2/TestLen/Seed0_20240725-133642'

# 绘制单个实验数据的图表
plot_single_experiment(single_csv_file, output_dir_single)
