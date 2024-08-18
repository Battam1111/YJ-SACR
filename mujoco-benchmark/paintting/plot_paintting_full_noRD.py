import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# 设置全局绘图样式
plt.style.use('seaborn-paper')

def read_csv_files(file_paths: List[str]) -> List[pd.DataFrame]:
    """读取CSV文件列表并返回DataFrame列表。对于包含'RefineT'的路径，忽略'deterministic'列。"""
    dfs = []
    for file in file_paths:
        if 'RefineT' in file:
            df = pd.read_csv(file).drop(columns=['deterministic'], errors='ignore')
        else:
            df = pd.read_csv(file)
        dfs.append(df)
    return dfs

def compute_mean_and_std(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算DataFrame列表的均值和标准差。"""
    combined_df = pd.concat(dfs)
    grouped = combined_df.groupby('epoch')
    mean_df = grouped.mean()
    std_df = grouped.std()
    return mean_df, std_df

def plot_data(dfs: List[pd.DataFrame], std_dfs: List[pd.DataFrame], labels: List[str], columns: List[str], xlabel: str, ylabel: str, title: str, output_file: str, window_size: int = 100, plot_std: bool = True):
    """绘制数据图表并保存到文件。"""
    plt.figure(figsize=(10, 6))
    for df, std_df, label in zip(dfs, std_dfs, labels):
        available_columns = [col for col in columns if col in df.columns]  # 过滤出存在的列
        for column in available_columns:
            plt.plot(df.index, df[column], label=f"{label} - {column} (Mean)")
            if plot_std:
                plt.fill_between(df.index, df[column] - std_df[column], df[column] + std_df[column], alpha=0.2)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    # 绘制平滑数据图
    plt.figure(figsize=(10, 6))
    for df, std_df, label in zip(dfs, std_dfs, labels):
        available_columns = [col for col in columns if col in df.columns]  # 过滤出存在的列
        for column in available_columns:
            smooth_mean = df[column].rolling(window=window_size).mean()
            if plot_std:
                smooth_std = std_df[column].rolling(window=window_size).mean()
                plt.fill_between(df.index, smooth_mean - smooth_std, smooth_mean + smooth_std, alpha=0.2)
            plt.plot(df.index, smooth_mean, label=f"{label} - {column} (Smoothed Mean)")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"Smoothed {title}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_smoothed.png'), dpi=300)
    plt.close()

def plot_single_data(dfs: List[pd.DataFrame], labels: List[str], columns: List[str], xlabel: str, ylabel: str, title: str, output_file: str, window_size: int = 100):
    """绘制单个数据图表并保存到文件。"""
    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels):
        available_columns = [col for col in columns if col in df.columns]  # 过滤出存在的列
        for column in available_columns:
            plt.plot(df.index, df[column], label=f"{label} - {column}")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    # 绘制平滑数据图
    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels):
        available_columns = [col for col in columns if col in df.columns]  # 过滤出存在的列
        for column in available_columns:
            smooth_data = df[column].rolling(window=window_size).mean()
            plt.plot(df.index, smooth_data, label=f"{label} - {column} (Smoothed)")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"Smoothed {title}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_smoothed.png'), dpi=300)
    plt.close()

def extract_info_from_path(file_path: str) -> Tuple[str, str, str, str]:
    """从文件路径中提取任务名称、AF参数、数据类型和种子。"""
    parts = file_path.split('/')
    task_name = parts[-4]
    af_param = parts[-3]
    data_type = parts[-2]
    seed = parts[-1].split('_')[0]
    return task_name, af_param, data_type, seed

def find_csv_files(root_dir: str) -> Tuple[List[str], List[str]]:
    """查找根目录下所有的TestRew和TestLen的CSV文件。"""
    csv_files_TR = []
    csv_files_TL = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                if 'TestRew' in dirpath:
                    csv_files_TR.append(os.path.join(dirpath, filename))
                elif 'TestLen' in dirpath:
                    csv_files_TL.append(os.path.join(dirpath, filename))
    return csv_files_TR, csv_files_TL

def plot_single_experiment(csv_files_TR1: List[str], csv_files_TL1: List[str], csv_files_TR2: List[str], csv_files_TL2: List[str], output_dir: str, af_param: str, seed: str, window_size: int = 100):
    """读取单个AF和种子组合的CSV文件并绘制图表。"""
    os.makedirs(output_dir, exist_ok=True)
    
    dfs_rewards1 = read_csv_files(csv_files_TR1)
    dfs_rewards2 = read_csv_files(csv_files_TR2)
    dfs_lengths1 = read_csv_files(csv_files_TL1)
    dfs_lengths2 = read_csv_files(csv_files_TL2)

    reward_columns = dfs_rewards1[0].columns[1:]
    length_columns = dfs_lengths1[0].columns[1:]

    reward_output_file = os.path.join(output_dir, f'AF({af_param})_Seed({seed})_average_reward.png')
    length_output_file = os.path.join(output_dir, f'AF({af_param})_Seed({seed})_episode_length.png')

    # 确保在绘制图表时正确区分Original和RefineT
    plot_single_data([dfs_rewards1[0], dfs_rewards2[0]], ["Original", "RefineT"], reward_columns, 'Epoch', 'Average Reward', f'Average Reward vs Epoch (AF={af_param}, Seed={seed})', reward_output_file, window_size)
    plot_single_data([dfs_lengths1[0], dfs_lengths2[0]], ["Original", "RefineT"], length_columns, 'Epoch', 'Episode Length', f'Episode Length vs Epoch (AF={af_param}, Seed={seed})', length_output_file, window_size)

def plot_multiple_experiments(csv_files_TR1: List[str], csv_files_TL1: List[str], csv_files_TR2: List[str], csv_files_TL2: List[str], output_dir: str, af_param: str, seeds: List[str], window_size: int = 100):
    """读取多个AF参数下的CSV文件，计算均值和标准差，并绘制图表。"""
    os.makedirs(output_dir, exist_ok=True)

    # 筛选出匹配的种子
    csv_files_TR1 = [file for file in csv_files_TR1 if extract_info_from_path(file)[3] in seeds]
    csv_files_TL1 = [file for file in csv_files_TL1 if extract_info_from_path(file)[3] in seeds]
    csv_files_TR2 = [file for file in csv_files_TR2 if extract_info_from_path(file)[3] in seeds]
    csv_files_TL2 = [file for file in csv_files_TL2 if extract_info_from_path(file)[3] in seeds]
    
    dfs_rewards1 = read_csv_files(csv_files_TR1)
    dfs_rewards2 = read_csv_files(csv_files_TR2)
    dfs_lengths1 = read_csv_files(csv_files_TL1)
    dfs_lengths2 = read_csv_files(csv_files_TL2)

    mean_rewards_df1, std_rewards_df1 = compute_mean_and_std(dfs_rewards1)
    mean_rewards_df2, std_rewards_df2 = compute_mean_and_std(dfs_rewards2)
    mean_lengths_df1, std_lengths_df1 = compute_mean_and_std(dfs_lengths1)
    mean_lengths_df2, std_lengths_df2 = compute_mean_and_std(dfs_lengths2)

    reward_columns = mean_rewards_df1.columns
    length_columns = mean_lengths_df1.columns

    seeds_str = '-'.join(seeds)

    reward_output_file = os.path.join(output_dir, f'AF({af_param})_Seeds({seeds_str})_average_reward.png')
    length_output_file = os.path.join(output_dir, f'AF({af_param})_Seeds({seeds_str})_episode_length.png')

    # 确保在绘制图表时正确区分Original和RefineT
    plot_data([mean_rewards_df1, mean_rewards_df2], [std_rewards_df1, std_rewards_df2], ["Original", "RefineT"], reward_columns, 'Epoch', 'Average Reward', f'Average Reward vs Epoch (AF={af_param})', reward_output_file, window_size)
    plot_data([mean_lengths_df1, mean_lengths_df2], [std_lengths_df1, std_lengths_df2], ["Original", "RefineT"], length_columns, 'Epoch', 'Episode Length', f'Episode Length vs Epoch (AF={af_param})', length_output_file, window_size)

def main(root_dir: str, output_root_dir: str, window_size: int = 100):
    """主函数，用于查找CSV文件并绘制图表。"""
    csv_files_TR, csv_files_TL = find_csv_files(root_dir)
    
    # 获取所有任务名称和AF参数
    tasks_and_params = set((extract_info_from_path(file)[:2]) for file in csv_files_TR + csv_files_TL)
    
    paired_tasks = {}
    for task, af_param in tasks_and_params:
        base_task = task.split('-')[0]
        if base_task not in paired_tasks:
            paired_tasks[base_task] = []
        paired_tasks[base_task].append((task, af_param))
    
    for base_task, pairs in paired_tasks.items():
        if len(pairs) == 0:
            print(f"任务 {base_task} 未找到任何配对的任务或参数")
            continue
        
        # 处理配对任务，确保原任务在前，配对任务在后
        original_tasks = [pair for pair in pairs if 'RefineT' not in pair[0]]
        refined_tasks = [pair for pair in pairs if 'RefineT' in pair[0]]
        
        for original_task in original_tasks:
            task1, af_param1 = original_task
            task1_csv_files_TR = [file for file in csv_files_TR if task1 in file and af_param1 in file]
            task1_csv_files_TL = [file for file in csv_files_TL if task1 in file and af_param1 in file]
            
            for refined_task in refined_tasks:
                task2, af_param2 = refined_task
                task2_csv_files_TR = [file for file in csv_files_TR if task2 in file and af_param2 in file]
                task2_csv_files_TL = [file for file in csv_files_TL if task2 in file and af_param2 in file]

                # 提取所有AF和种子组合
                all_af_seeds = set()
                for file in task1_csv_files_TR + task2_csv_files_TR + task1_csv_files_TL + task2_csv_files_TL:
                    _, af_param, _, seed = extract_info_from_path(file)
                    all_af_seeds.add((af_param, seed))
                
                # 单个AF和种子组合的图表
                for af_param, seed in all_af_seeds:
                    single_dir = os.path.join(output_root_dir, base_task, 'single', f'AF({af_param})_Seed({seed})')
                    
                    # 只包含原任务的文件
                    csv_files_TR1 = [file for file in task1_csv_files_TR if af_param in file and seed in file and 'RefineT' not in file]
                    csv_files_TL1 = [file for file in task1_csv_files_TL if af_param in file and seed in file and 'RefineT' not in file]
                    
                    # 只包含配对任务的文件
                    csv_files_TR2 = [file for file in task2_csv_files_TR if af_param in file and seed in file and 'RefineT' in file]
                    csv_files_TL2 = [file for file in task2_csv_files_TL if af_param in file and seed in file and 'RefineT' in file]

                    if csv_files_TR1 and csv_files_TL1 and csv_files_TR2 and csv_files_TL2:
                        plot_single_experiment(csv_files_TR1, csv_files_TL1, csv_files_TR2, csv_files_TL2, single_dir, af_param, seed, window_size)
                    else:
                        print(f"任务 {base_task} 的 AF={af_param} 和 Seed={seed} 缺少文件")

                # 多个AF参数组合的图表
                all_af_params = set(af_param for af_param, _ in all_af_seeds)
                for af_param in all_af_params:
                    # 找出共同的种子
                    seeds1 = set(extract_info_from_path(f)[3] for f in task1_csv_files_TR if af_param in f)
                    seeds2 = set(extract_info_from_path(f)[3] for f in task2_csv_files_TR if af_param in f)
                    common_seeds = list(seeds1 & seeds2)

                    if common_seeds:
                        multiple_dir = os.path.join(output_root_dir, base_task, 'multiple', f'AF({af_param})')
                        csv_files_TR1 = [file for file in task1_csv_files_TR if af_param in file and extract_info_from_path(file)[3] in common_seeds]
                        csv_files_TL1 = [file for file in task1_csv_files_TL if af_param in file and extract_info_from_path(file)[3] in common_seeds]
                        csv_files_TR2 = [file for file in csv_files_TR if task2 in file and af_param2 in file and extract_info_from_path(file)[3] in common_seeds]
                        csv_files_TL2 = [file for file in csv_files_TL if task2 in file and af_param2 in file and extract_info_from_path(file)[3] in common_seeds]

                        plot_multiple_experiments(csv_files_TR1, csv_files_TL1, csv_files_TR2, csv_files_TL2, multiple_dir, af_param, common_seeds, window_size)
                    else:
                        print(f"任务 {base_task} 的 AF={af_param} 没有共同的种子")


# 使用示例
root_dir = 'mujoco-benchmark/data'
output_root_dir = 'mujoco-benchmark/PlotSet-FULL-NoRD'
main(root_dir, output_root_dir)
