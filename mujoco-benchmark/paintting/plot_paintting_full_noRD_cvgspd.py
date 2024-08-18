import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# 设置全局绘图样式
plt.style.use('seaborn-paper')

def read_csv_files(file_paths: List[str]) -> List[pd.DataFrame]:
    """读取CSV文件列表，返回DataFrame列表，自动忽略'RefineT'文件中的'deterministic'列。"""
    return [pd.read_csv(file).drop(columns=['deterministic'], errors='ignore') if 'RefineT' in file else pd.read_csv(file) for file in file_paths]

def compute_mean_and_std(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算DataFrame列表按'epoch'分组后的均值和标准差。"""
    combined_df = pd.concat(dfs)
    grouped = combined_df.groupby('epoch')
    return grouped.mean(), grouped.std()

# def compute_convergence_speed(df: pd.DataFrame, column: str) -> float:
#     """
#     计算收敛速度：定义为达到最大奖励90%时的训练步数。
    
#     :param df: 包含奖励数据的DataFrame
#     :param column: 需要计算的列名
#     :return: 达到90%最大奖励时的训练步数
#     """
#     max_reward = df[column].max()
#     threshold = 0.9 * max_reward
    
#     # 找到首次达到或超过90%最大奖励的epoch
#     for epoch, reward in enumerate(df[column]):
#         if reward >= threshold:
#             return epoch + 1  # 返回epoch加1，因为epoch是从0开始计数的
    
#     # 如果在所有epoch中都没有达到阈值，返回总epoch数
#     return len(df)

def compute_convergence_speed(df: pd.DataFrame, column: str) -> float:
    """计算给定DataFrame列的收敛速度，作为每个epoch差值的平均值。"""
    diffs = df[column].diff().fillna(0)
    return diffs.mean()

def plot_convergence_speed_bar_chart(convergence_speeds: dict, label_color_map: dict, output_file: str, label_fontsize: int = 15):
    """
    绘制收敛速度的柱状图。
    :param convergence_speeds: 各个数据标签对应的收敛速度字典
    :param label_color_map: 数据标签对应的颜色映射
    :param output_file: 保存柱状图的文件路径
    :param label_fontsize: 控制X轴标签的字体大小
    """
    labels = list(convergence_speeds.keys())
    speeds = [convergence_speeds[label] for label in labels]
    colors = [label_color_map[label] for label in labels]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, speeds, color=colors)
    plt.ylabel('Convergence Speed', fontsize=17)
    plt.xticks(fontsize=label_fontsize)  # 设置X轴标签的字体大小
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)  # 保存为PDF格式
    plt.close()

def plot_data(dfs: List[pd.DataFrame], std_dfs: List[pd.DataFrame], labels: List[str], columns: List[str], xlabel: str, ylabel: str, output_file: str, window_size: int = 100, plot_std: bool = True):
    """绘制数据图表、平滑图和滑动平滑图，并保存到文件。"""
    label_color_map = {
        "Original - refineSampling": "#1f77b4",
        "RefineT - refineSampling": "#ff7f0e",
        "Original - deterministic": "#2ca02c",
    }

    # 初始化存储收敛速度的字典
    convergence_speeds = {}

    def plot_curve(suffix: str, smooth: bool = False, sliding: bool = False, plot_convergence_speed: bool = False):
        """内部函数，用于绘制曲线，并可选地绘制收敛速度。"""
        plt.figure(figsize=(10, 6))
        for df, std_df, label in zip(dfs, std_dfs, labels):
            available_columns = [col for col in columns if col in df.columns]
            for column in available_columns:
                data = df[column].rolling(window=window_size).mean() if smooth else df[column]
                if sliding:
                    data = data.rolling(window=window_size).mean()
                color = label_color_map.get(f"{label} - {column}", "black")  # 默认颜色为黑色
                
                # 绘制数据曲线
                plt.plot(df.index, data, label=f"{label} - {column}", color=color)
                
                # 绘制标准差带
                if plot_std:
                    std_data = std_df[column].rolling(window=window_size).mean() if smooth else std_df[column]
                    if sliding:
                        std_data = std_data.rolling(window=window_size).mean()
                    plt.fill_between(df.index, data - std_data, data + std_data, alpha=0.2, color=color)
                
                # 计算并存储收敛速度
                if plot_convergence_speed:
                    convergence_speed = compute_convergence_speed(df, column)
                    convergence_speeds[f"{label} - {column}"] = convergence_speed

        plt.xlabel(xlabel, fontsize=17)
        plt.ylabel(ylabel, fontsize=17)
        
        # 调整图例的位置到图像的正上方，确保只有一行，手动超参3项数据
        plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)
        
        plt.grid(True)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 调整顶部间距以适应图例
        plt.savefig(output_file.replace('.pdf', f'{suffix}.pdf'), dpi=300)  # 保存为PDF格式
        plt.close()

        # 如果需要绘制收敛速度的柱状图
        if plot_convergence_speed and convergence_speeds:
            bar_chart_output_file = output_file.replace('.pdf', f'{suffix}_bar_chart.pdf')
            plot_convergence_speed_bar_chart(convergence_speeds, label_color_map, bar_chart_output_file)

    # 绘制原始数据图
    plot_curve(suffix='_raw')

    # 绘制平滑数据图
    plot_curve(suffix='_smoothed', smooth=True)

    # 绘制滑动平滑数据图
    plot_curve(suffix='_sliding', smooth=True, sliding=True)

    # 绘制包含收敛速度的图，并绘制收敛速度的柱状图
    plot_curve(suffix='_convergence_speed', plot_convergence_speed=True)

def extract_info_from_path(file_path: str) -> Tuple[str, str, str, str]:
    """从文件路径中提取任务名称、AF参数、数据类型和种子。"""
    parts = file_path.split('/')
    return parts[-4], parts[-3], parts[-2], parts[-1].split('_')[0]

def find_csv_files(root_dir: str) -> Tuple[List[str], List[str]]:
    """查找根目录下所有的TestRew和TestLen的CSV文件。"""
    csv_files_TR, csv_files_TL = [], []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                full_path = os.path.join(dirpath, filename)
                if 'TestRew' in dirpath:
                    csv_files_TR.append(full_path)
                elif 'TestLen' in dirpath:
                    csv_files_TL.append(full_path)
    return csv_files_TR, csv_files_TL

def plot_single_experiment(csv_files_TR1: List[str], csv_files_TL1: List[str], csv_files_TR2: List[str], csv_files_TL2: List[str], output_dir: str, af_param: str, seed: str, window_size: int = 100):
    """读取单个AF和种子组合的CSV文件并绘制图表。"""
    os.makedirs(output_dir, exist_ok=True)

    # 剔除csv_files_TR1和csv_files_TL1列表中所有包含“RefineT”字符的元素
    csv_files_TR1 = [file for file in csv_files_TR1 if 'RefineT' not in file]
    csv_files_TL1 = [file for file in csv_files_TL1 if 'RefineT' not in file]
    
    # 确保csv_files_TR2和csv_files_TL2列表中所有元素包含“RefineT”字符
    csv_files_TR2 = [file for file in csv_files_TR2 if 'RefineT' in file]
    csv_files_TL2 = [file for file in csv_files_TL2 if 'RefineT' in file]

    dfs_rewards1, dfs_rewards2 = read_csv_files(csv_files_TR1), read_csv_files(csv_files_TR2)
    dfs_lengths1, dfs_lengths2 = read_csv_files(csv_files_TL1), read_csv_files(csv_files_TL2)
    
    # 计算均值和标准差
    mean_rewards_df1, std_rewards_df1 = compute_mean_and_std(dfs_rewards1)
    mean_rewards_df2, std_rewards_df2 = compute_mean_and_std(dfs_rewards2)
    mean_lengths_df1, std_lengths_df1 = compute_mean_and_std(dfs_lengths1)
    mean_lengths_df2, std_lengths_df2 = compute_mean_and_std(dfs_lengths2)

    reward_columns = mean_rewards_df1.columns  # 直接使用均值数据的列
    length_columns = mean_lengths_df1.columns

    # 确保labels与数据的顺序对应
    plot_data([mean_rewards_df1, mean_rewards_df2], [std_rewards_df1, std_rewards_df2], ["Original", "RefineT"], reward_columns, 'Epoch', 'Average Reward', os.path.join(output_dir, f'AF({af_param})_average_reward.pdf'), window_size)
    # plot_data([mean_rewards_df1, mean_rewards_df2], [std_rewards_df1, std_rewards_df2], ["Original", "RefineT"], reward_columns, 'Epoch', 'Average Reward', os.path.join(output_dir, f'AF({af_param})_Seed({seed})_average_reward.pdf'), window_size)
    # plot_data([mean_lengths_df1, mean_lengths_df2], [std_rewards_df1, std_rewards_df2], ["Original", "RefineT"], length_columns, 'Epoch', 'Episode Length', os.path.join(output_dir, f'AF({af_param})_Seed({seed})_episode_length.pdf'), window_size)

def plot_multiple_experiments(csv_files_TR1: List[str], csv_files_TL1: List[str], csv_files_TR2: List[str], csv_files_TL2: List[str], output_dir: str, af_param: str, seeds: List[str], window_size: int = 100):
    """读取多个AF参数下的CSV文件，计算均值和标准差，并绘制图表。"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 剔除csv_files_TR1和csv_files_TL1列表中所有包含“RefineT”字符的元素
    csv_files_TR1 = [file for file in csv_files_TR1 if 'RefineT' not in file]
    csv_files_TL1 = [file for file in csv_files_TL1 if 'RefineT' not in file]
    
    # 确保csv_files_TR2和csv_files_TL2列表中所有元素包含“RefineT”字符
    csv_files_TR2 = [file for file in csv_files_TR2 if 'RefineT' in file]
    csv_files_TL2 = [file for file in csv_files_TL2 if 'RefineT' in file]

    # 筛选并读取文件
    dfs_rewards1 = read_csv_files([file for file in csv_files_TR1 if extract_info_from_path(file)[3] in seeds])
    dfs_rewards2 = read_csv_files([file for file in csv_files_TR2 if extract_info_from_path(file)[3] in seeds])
    dfs_lengths1 = read_csv_files([file for file in csv_files_TL1 if extract_info_from_path(file)[3] in seeds])
    dfs_lengths2 = read_csv_files([file for file in csv_files_TL2 if extract_info_from_path(file)[3] in seeds])
    
    # 计算均值和标准差
    mean_rewards_df1, std_rewards_df1 = compute_mean_and_std(dfs_rewards1)
    mean_rewards_df2, std_rewards_df2 = compute_mean_and_std(dfs_rewards2)
    mean_lengths_df1, std_lengths_df1 = compute_mean_and_std(dfs_lengths1)
    mean_lengths_df2, std_lengths_df2 = compute_mean_and_std(dfs_lengths2)

    seeds_str = '-'.join(seeds)

    plot_data([mean_rewards_df1, mean_rewards_df2], [std_rewards_df1, std_rewards_df2], ["Original", "RefineT"], mean_rewards_df1.columns, 'Epoch', 'Average Reward', os.path.join(output_dir, f'AF({af_param})_average_reward.pdf'), window_size)
    # plot_data([mean_rewards_df1, mean_rewards_df2], [std_rewards_df1, std_rewards_df2], ["Original", "RefineT"], mean_rewards_df1.columns, 'Epoch', 'Average Reward', os.path.join(output_dir, f'AF({af_param})_Seeds({seeds_str})_average_reward.pdf'), window_size)
    # plot_data([mean_lengths_df1, mean_lengths_df2], [std_rewards_df1, std_rewards_df2], ["Original", "RefineT"], mean_lengths_df1.columns, 'Epoch', 'Episode Length', os.path.join(output_dir, f'AF({af_param})_Seeds({seeds_str})_episode_length.pdf'), window_size)

def main(root_dir: str, output_root_dir: str, window_size: int = 10):
    """主函数，用于查找CSV文件并绘制图表。"""
    csv_files_TR, csv_files_TL = find_csv_files(root_dir)
    
    tasks_and_params = {(extract_info_from_path(file)[:2]) for file in csv_files_TR + csv_files_TL}
    paired_tasks = {}
    
    for task, af_param in tasks_and_params:
        base_task = task.split('-')[0]
        paired_tasks.setdefault(base_task, []).append((task, af_param))
    
    for base_task, pairs in paired_tasks.items():
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

                all_af_seeds = {(extract_info_from_path(f)[1], extract_info_from_path(f)[3]) for f in task1_csv_files_TR + task2_csv_files_TR + task1_csv_files_TL + task2_csv_files_TL}

                for af_param, seed in all_af_seeds:
                    single_dir = os.path.join(output_root_dir, base_task, 'single', f'AF({af_param})_Seed({seed})')
                    csv_files_TR1 = [f for f in task1_csv_files_TR if af_param in f and seed in f and 'RefineT' not in f]
                    csv_files_TL1 = [f for f in task1_csv_files_TL if af_param in f and seed in f and 'RefineT' not in f]
                    csv_files_TR2 = [f for f in task2_csv_files_TR if af_param in f and seed in f and 'RefineT' in f]
                    csv_files_TL2 = [f for f in task2_csv_files_TL if af_param in f and seed in f and 'RefineT' in f]

                    # 不跑单个实验的了
                    # if csv_files_TR1 and csv_files_TL1 and csv_files_TR2 and csv_files_TL2:
                    #     plot_single_experiment(csv_files_TR1, csv_files_TL1, csv_files_TR2, csv_files_TL2, single_dir, af_param, seed, window_size)
                    # else:
                    #     print(f"任务 {base_task} 的 AF={af_param} 和 Seed={seed} 缺少文件")

                all_af_params = {af_param for af_param, _ in all_af_seeds}
                for af_param in all_af_params:
                    seeds1 = {extract_info_from_path(f)[3] for f in task1_csv_files_TR if af_param in f}
                    seeds2 = {extract_info_from_path(f)[3] for f in task2_csv_files_TR if af_param in f}
                    common_seeds = list(seeds1 & seeds2)

                    if common_seeds:
                        multiple_dir = os.path.join(output_root_dir, base_task, 'multiple', f'AF({af_param})')
                        csv_files_TR1 = [f for f in task1_csv_files_TR if af_param in f and extract_info_from_path(f)[3] in common_seeds]
                        csv_files_TL1 = [f for f in task1_csv_files_TL if af_param in f and extract_info_from_path(f)[3] in common_seeds]
                        csv_files_TR2 = [f for f in csv_files_TR if task2 in f and af_param2 in f and extract_info_from_path(f)[3] in common_seeds]
                        csv_files_TL2 = [f for f in csv_files_TL if task2 in f and af_param2 in f and extract_info_from_path(f)[3] in common_seeds]

                        plot_multiple_experiments(csv_files_TR1, csv_files_TL1, csv_files_TR2, csv_files_TL2, multiple_dir, af_param, common_seeds, window_size)
                    else:
                        print(f"任务 {base_task} 的 AF={af_param} 没有共同的种子")

# 使用示例
root_dir = 'mujoco-benchmark/data'
output_root_dir = 'mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd'
main(root_dir, output_root_dir)
