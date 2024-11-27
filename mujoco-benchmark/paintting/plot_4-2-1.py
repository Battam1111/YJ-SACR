import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import re
import seaborn as sns

# 设置全局绘图样式
sns.set(style='white')

# 定义颜色常量
BASELINE_COLOR = '#498DD1'   # 蓝色，用于 Standard
ERROR_COLOR = '#ff7f0e'  # 橙色，用于 错误点
CORRECTED_COLOR = '#E15E39'     # 红色，用于 Corrected

def read_csv_files(file_paths: List[str]) -> List[pd.DataFrame]:
    """读取CSV文件列表，返回DataFrame列表，并添加'Seed'列。"""
    dfs = []
    for file in file_paths:
        df = pd.read_csv(file)
        seed_value = extract_seed_from_filename(file)
        df['Seed'] = seed_value
        dfs.append(df)
    return dfs

def compute_mean_and_std(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算DataFrame列表按'Step'分组后的均值和标准差。"""
    combined_df = pd.concat(dfs)
    # Convert columns to appropriate dtypes
    for col in ['Step', 'Corrected', 'Standard']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    # Optionally drop rows with NaN values in key columns
    combined_df.dropna(subset=['Step', 'Corrected', 'Standard'], inplace=True)
    # Exclude non-numeric columns
    numeric_cols = ['Corrected', 'Standard']
    grouped = combined_df.groupby('Step')[numeric_cols]
    mean_df = grouped.mean()
    std_df = grouped.std()
    return mean_df, std_df

def plot_data(mean_df: pd.DataFrame, std_df: pd.DataFrame, columns: List[str], xlabel: str, ylabel: str, title: str, output_file: str, window_size: int = 100, plot_std: bool = True):
    """绘制数据图表，并保存到文件。"""
    plt.figure(figsize=(10, 6))
    
    # 定义颜色映射
    style_map = {
        'Standard': {
            'color': BASELINE_COLOR,# 蓝色
            'alpha': 1.0,  # 主曲线透明度
            'fill_alpha': 0.15,  # 填充区域透明度
            'linewidth': 1.5,
            'fill_linewidth': 1.0
        },
        'Corrected': {
            'color': CORRECTED_COLOR,# 红色
            'alpha': 1.0,  # 主曲线透明度
            'fill_alpha': 0.3,  # 填充区域透明度
            'linewidth': 1.5,
            'fill_linewidth': 1.0
        }
    }    
    for column in columns:
        data = mean_df[column]
        std_data = std_df[column]
        # Apply rolling mean for smoothing if needed
        data_smoothed = data.rolling(window=window_size, min_periods=1).mean()
        std_data_smoothed = std_data.rolling(window=window_size, min_periods=1).mean()
        # 主曲线
        plt.plot(mean_df.index[0:-1:3], 
                 data_smoothed[0:-1:3], 
                 label=f'{column}', 
                 color=style_map[column]['color'],
                 alpha=style_map[column]['alpha'],
                 linewidth=style_map[column]['linewidth'])
        # 阴影
        if plot_std:
            plt.fill_between(mean_df.index[0:-1:3],
                             data_smoothed[0:-1:3] - std_data_smoothed[0:-1:3],
                             data_smoothed[0:-1:3] + std_data_smoothed[0:-1:3],
                             color=style_map[column]['color'],
                             alpha=style_map[column]['fill_alpha'],
                             linewidth=style_map[column]['fill_linewidth'])
    
    # 设置标题和标签的字体大小和加粗
    plt.xlabel(xlabel, fontsize=18, fontweight='bold')
    plt.ylabel(ylabel, fontsize=18, fontweight='bold')
    # plt.title(title, fontsize=18, fontweight='bold')

    # 设置坐标轴的字体大小和加粗
    plt.tick_params(axis='both', labelsize=16, width=2)

    # 加粗边框轮廓
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)  # 设置边框线宽

    # 绘制图例，并调整字体
    legend = plt.legend(fontsize=16,loc='upper left')  # 设置字体大小
    for label in legend.get_texts():
        label.set_fontweight('bold')  # 设置字体加粗

    # 清空 x 轴偏移字符串，保留科学计数法
    plt.gca().xaxis.get_offset_text().set_visible(False)

    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_file, dpi=600)
    plt.close()



def extract_seed_from_filename(filename: str) -> str:
    """从文件名中提取种子值，使用正则表达式匹配。"""
    basename = os.path.basename(filename)
    # Pattern to match '_seed_<seed_value>'
    match = re.search(r'_seed_(\d+)', basename)
    if match:
        seed_value = match.group(1)
    else:
        # Attempt to extract the last number in the filename as seed
        numbers = re.findall(r'(\d+)', basename)
        if numbers:
            seed_value = numbers[-1]
        else:
            print(f"No seed value found in filename {basename}. Assigning default seed value.")
            seed_value = '0'  # Default seed value or handle accordingly
    return seed_value

def get_tasks(root_dir: str) -> List[str]:
    """获取根目录下的所有任务名称（子文件夹名称）。"""
    tasks = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    return tasks

def process_task(task_dir: str, output_dir: str, window_size: int = 100):
    """处理单个任务，读取CSV文件，计算均值和标准差，并绘制图表。"""
    csv_files = [os.path.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"在 {task_dir} 中未找到CSV文件")
        return

    # 读取CSV文件并添加'Seed'列
    dfs = read_csv_files(csv_files)

    # 计算均值和标准差
    mean_df, std_df = compute_mean_and_std(dfs)

    # 要绘制的列
    columns = ['Corrected', 'Standard']

    # 绘制图表
    xlabel = f'Environment Steps ($10^7$)'
    ylabel = 'Return'

    # xlabel = ' '
    # ylabel = ' '

    title = f"{os.path.basename(task_dir)} - Performance Over Steps"
    output_file = os.path.join(output_dir, f"{os.path.basename(task_dir)}.pdf")
    plot_data(mean_df, std_df, columns, xlabel, ylabel, title, output_file, window_size)

def main(root_dir: str, output_root_dir: str, window_size: int = 10):
    """主函数，遍历所有任务并绘制图表。"""
    tasks = get_tasks(root_dir)
    for task in tasks:
        task_dir = os.path.join(root_dir, task)
        output_dir = os.path.join(output_root_dir, task)
        os.makedirs(output_dir, exist_ok=True)
        process_task(task_dir, output_dir, window_size)

# 使用示例
if __name__ == '__main__':
    root_dir = 'mujoco-benchmark/data-humanbench'  # 请根据实际路径修改
    output_root_dir = 'mujoco-benchmark/paintting/humanoidbench-paintting/plot_4-2/plot_4-2-1'  # 请根据实际路径修改
    main(root_dir, output_root_dir)
