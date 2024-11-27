import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rliable import library as rly, metrics
from rliable import plot_utils
import seaborn as sns

# 设置全局样式和字体
def configure_plot_settings():
    """
    配置全局的图像样式和参数，便于全局控制图像的样式。
    """
    sns.set(style='white')
    plt.rcParams.update({
        'font.size': 12,
        'figure.dpi': 300,
        'figure.figsize': (12, 8),  # 默认图像尺寸
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

class RLDataProcessor:
    """
    数据处理类，负责加载和清洗数据，并确保步长的一致性。
    """
    def __init__(self, data_path):
        """
        初始化数据处理类。
        :param data_path: 数据的根目录。
        """
        self.data_path = data_path
        self.tasks_data = {}
        self.steps = None

    def load_data(self):
        """
        加载每个任务的数据，确保步长一致。
        """
        for task_folder in glob.glob(os.path.join(self.data_path, "*")):
            task_name = os.path.basename(task_folder)
            task_baseline, task_corrected = [], []

            for file in glob.glob(os.path.join(task_folder, "*.csv")):
                df = pd.read_csv(file)
                if not self.validate_and_extract_steps(df, file):
                    continue
                task_baseline.append(df["Standard"].values)
                task_corrected.append(df["Corrected"].values)

            if task_baseline:
                self.tasks_data[task_name] = {
                    "Standard": np.array(task_baseline, dtype=object),
                    "Corrected": np.array(task_corrected, dtype=object)
                }
            else:
                print(f"Warning: No valid data found for task {task_name}, skipping.")

    def validate_and_extract_steps(self, df, filename):
        """
        验证步长一致性。
        :param df: DataFrame
        :param filename: 文件名（用于错误提示）
        :return: 是否步长一致
        """
        if self.steps is None:
            self.steps = df["Step"].values
        elif not np.array_equal(self.steps, df["Step"].values):
            print(f"Warning: Skipping {filename} due to mismatched steps.")
            return False
        return True

    @staticmethod
    def clean_data(data):
        """
        清洗数据，移除 NaN 值。
        :param data: 原始数据
        :return: 清洗后的数据
        """
        cleaned_data = np.array(data, dtype=np.float64)
        cleaned_data = cleaned_data[~np.isnan(cleaned_data).any(axis=1)]
        return cleaned_data

class RLPlotter:
    """
    画图类，负责基于处理后的数据生成图表。
    """
    
    # 定义颜色常量
    BASELINE_COLOR = '#498DD1'   # 蓝色，用于 Standard
    ERROR_COLOR = '#ff7f0e'  # 橙色，用于 错误点
    CORRECTED_COLOR = '#E15E39'     # 红色，用于 Corrected

    def __init__(self, output_path, processor):
        """
        初始化画图类。
        :param output_path: 图像保存路径
        :param processor: RLDataProcessor 实例，用于数据获取和清洗
        """
        self.output_path = output_path
        self.processor = processor
        self.ensure_output_dirs()

    def ensure_output_dirs(self):
        """
        创建图像保存目录。
        """
        plot_types = ["aggregate_metrics", "performance_profiles", "probability_of_improvement", "sample_efficiency"]
        for plot_type in plot_types:
            os.makedirs(os.path.join(self.output_path, plot_type), exist_ok=True)

    @staticmethod
    def smooth_curve(data, window_size=100):
        """
        使用滚动窗口平滑数据。
        :param data: 输入数据
        :param window_size: 平滑窗口大小
        :return: 平滑后的数据
        """
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

    def plot_bar_with_error(self, ax, x, y, yerr, label, color, width=0.35):
        """
        绘制带误差条的条形图，误差条包含更显著的横杠（end caps）。
        """
        # 保证误差条的终端有显著的横杠（end caps）
        error_kw = dict(ecolor='black', elinewidth=1.5, capsize=10, capthick=1)
        yerr = [np.maximum(0, val) for val in yerr]  # 修正负值的误差条
        ax.bar(x, y, width=width, label=label, color=color, yerr=yerr, capsize=10, alpha=0.7,
               error_kw=error_kw)
        
    def plot_with_confidence_interval(self, ax, x, y, ci_lower, ci_upper, label, color):
        """
        绘制带置信区间的曲线图。
        """
        ax.plot(x, y, label=label, color=color, linewidth=2)
        ax.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)

    def compute_aggregate_metrics(self, task_name, baseline_data, corrected_data):
        """
        计算并绘制聚合指标的条形图，带有置信区间。
        """
        # 定义聚合指标计算函数
        aggregate_func = lambda x: np.array([
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x)
        ])

        # 获取基线和校正数据的聚合指标和置信区间
        baseline_scores, baseline_cis = rly.get_interval_estimates(
            {'Standard': baseline_data}, aggregate_func, reps=500)
        corrected_scores, corrected_cis = rly.get_interval_estimates(
            {'Corrected': corrected_data}, aggregate_func, reps=500)

        # 设置图像参数
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(3)  # 横坐标位置，忽略 Optimality Gap

        # 绘制Baseline和Corrected的条形图
        self.plot_bar_with_error(
            ax, x - 0.2, baseline_scores['Standard'][:3], 
            yerr=[baseline_scores['Standard'][:3] - baseline_cis['Standard'][0][:3], baseline_cis['Standard'][1][:3] - baseline_scores['Standard'][:3]], 
            label='Standard', color=self.BASELINE_COLOR
        )
        self.plot_bar_with_error(
            ax, x + 0.2, corrected_scores['Corrected'][:3], 
            yerr=[corrected_scores['Corrected'][:3] - corrected_cis['Corrected'][0][:3], corrected_cis['Corrected'][1][:3] - corrected_scores['Corrected'][:3]], 
            label='Corrected', color=self.CORRECTED_COLOR
        )

        # 美化图表
        ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')  # 加大并加粗
        ax.set_ylabel('Performance Score', fontsize=16, fontweight='bold')  # 加大并加粗
        ax.set_title(f"Aggregate Metrics for {task_name}", fontsize=18, fontweight='bold')  # 加大并加粗
        ax.set_xticks(x)
        ax.set_xticklabels(['Median', 'IQM', 'Mean'], fontsize=14, fontweight='bold')  # 加大并加粗

        # 绘制图例，并调整字体
        legend = ax.legend(loc='upper left', fontsize=14)  # 设置字体大小
        for label in legend.get_texts():
            label.set_fontweight('bold')  # 设置字体加粗

        # ax.legend(loc='upper left', fontsize=14, fontweight='bold')  # 加大并加粗
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 加粗坐标轴边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)  # 加粗左边框
        ax.spines['bottom'].set_linewidth(2)  # 加粗底边框

        # 保存图表
        save_path = os.path.join(self.output_path, "aggregate_metrics", f"{task_name}_aggregate_metrics.pdf")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)



    def compute_performance_profiles(self, task_name, baseline_data, corrected_data):
        """
        计算并绘制性能分布曲线，添加标准差阴影区域。
        """
        # 根据任务进行选择性设置
        if task_name == "h1hand-bookshelf_hard-v0":
            score_thresholds = np.linspace(200, 400, 50)
        elif task_name == "h1hand-bookshelf_simple-v0":
            score_thresholds = np.linspace(200, 500, 50)
        elif task_name == "h1hand-cube-v0":
            score_thresholds = np.linspace(80, 140, 50)
        elif task_name == "h1hand-powerlift-v0":
            score_thresholds = np.linspace(110, 140, 50)

        # 获取Baseline和Corrected的分布和置信区间
        baseline_distributions, baseline_cis = rly.create_performance_profile(
            {'Standard': baseline_data}, score_thresholds)
        corrected_distributions, corrected_cis = rly.create_performance_profile(
            {'Corrected': corrected_data}, score_thresholds)

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制Baseline和Corrected曲线
        self.plot_with_confidence_interval(
            ax, score_thresholds, baseline_distributions['Standard'],
            baseline_cis['Standard'][0], baseline_cis['Standard'][1],
            label='Standard', color=self.BASELINE_COLOR
        )
        self.plot_with_confidence_interval(
            ax, score_thresholds, corrected_distributions['Corrected'],
            corrected_cis['Corrected'][0], corrected_cis['Corrected'][1],
            label='Corrected', color=self.CORRECTED_COLOR
        )

        # 设置标题和标签
        ax.set_title(f"Performance Profiles for {task_name}", fontsize=18, fontweight='bold')
        ax.set_xlabel('Performance Threshold', fontsize=16, fontweight='bold')
        ax.set_ylabel('Proportion of Runs', fontsize=16, fontweight='bold')
        ax.set_xlim(score_thresholds.min(), score_thresholds.max())
        ax.set_ylim(0, 1)
        # ax.legend(loc='upper left', fontsize=12)
        # 绘制图例，并调整字体
        legend = ax.legend(loc='upper left', fontsize=14)  # 设置字体大小
        for label in legend.get_texts():
            label.set_fontweight('bold')  # 设置字体加粗
        ax.grid(axis='both', linestyle='--', alpha=0.6)


        # 加粗坐标轴边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)  # 加粗左边框
        ax.spines['bottom'].set_linewidth(2)  # 加粗底边框

        # 保存图表
        save_path = os.path.join(self.output_path, "performance_profiles", f"{task_name}_performance_profiles.pdf")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)

    def compute_probability_of_improvement(self, task_name, baseline_data, corrected_data):
        """
        计算并绘制改进概率图，显示 Corrected 相对于 Standard 算法的比较结果。
        """
        # 清洗数据，去除 NaN
        baseline_data = self.processor.clean_data(baseline_data)
        corrected_data = self.processor.clean_data(corrected_data)

        # 验证数据的维度是否满足绘图需求
        if baseline_data.ndim != 2 or corrected_data.ndim != 2:
            print(f"Skipping probability of improvement for {task_name} due to insufficient data dimensions.")
            return

        # 计算改进概率和置信区间
        algorithm_pairs = {'Corrected,Standard': (corrected_data, baseline_data)}
        average_probabilities, average_prob_cis = rly.get_interval_estimates(
            algorithm_pairs, metrics.probability_of_improvement, reps=500
        )

        # 提取改进概率和置信区间
        prob_improvement = average_probabilities['Corrected,Standard']
        prob_ci_lower, prob_ci_upper = average_prob_cis['Corrected,Standard']

        # 调整误差条形状
        xerr = np.array([[prob_improvement - prob_ci_lower], [prob_ci_upper - prob_improvement]]).reshape(2, 1)

        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 5))

        # 绘制Corrected的点，并添加误差条
        ax.errorbar(
            x=[prob_improvement], y=[0], xerr=xerr,  # 只设置xerr，确保误差条在水平方向
            fmt='o', color=self.CORRECTED_COLOR, ecolor='lightgray', capsize=5, elinewidth=2, markeredgewidth=2, label='Corrected'
        )

        # 绘制Standard的点
        ax.plot([0.5], [0], 'o', color=self.BASELINE_COLOR, label='Standard', markersize=8)

        # 设置标题和标签
        ax.set_title(f"Probability of Improvement for {task_name}", fontsize=18, fontweight='bold')
        ax.set_xlabel("P(Corrected > Standard)", fontsize=16, fontweight='bold')
        ax.set_yticks([])  # 隐藏y轴刻度

        # 设置图例和x轴
        # ax.legend(fontsize=12)
        # 绘制图例，并调整字体
        legend = ax.legend(loc='upper left', fontsize=14)  # 设置字体大小
        for label in legend.get_texts():
            label.set_fontweight('bold')  # 设置字体加粗
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))

        # 添加网格和边框调整
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        # 加粗坐标轴边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)  # 加粗左边框
        ax.spines['bottom'].set_linewidth(2)  # 加粗底边框

        # 保存图表
        save_path = os.path.join(self.output_path, "probability_of_improvement", f"{task_name}_probability_of_improvement.pdf")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)

    def compute_sample_efficiency_curve(self, task_name, baseline_data, corrected_data):
        """
        计算并绘制样本效率曲线，带有标准差阴影区域，显示Baseline和Corrected的表现。
        """
        # 定义训练帧数
        frames = np.arange(0, baseline_data.shape[1], 1)

        # 定义IQM函数
        def iqm_func(scores):
            return np.array([metrics.aggregate_iqm(scores[:, i]) for i in frames])

        # 计算Baseline和Corrected的IQM表现及置信区间
        baseline_iqm_scores, baseline_iqm_cis = rly.get_interval_estimates(
            {'Standard': baseline_data}, iqm_func, reps=500)
        corrected_iqm_scores, corrected_cis = rly.get_interval_estimates(
            {'Corrected': corrected_data}, iqm_func, reps=500)

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制Baseline曲线并填充置信区间
        self.plot_with_confidence_interval(
            ax, frames + 1, self.smooth_curve(baseline_iqm_scores['Standard']),
            baseline_iqm_cis['Standard'][0], baseline_iqm_cis['Standard'][1],
            label='Standard', color=self.BASELINE_COLOR
        )

        # 绘制Corrected曲线并填充置信区间
        self.plot_with_confidence_interval(
            ax, frames + 1, self.smooth_curve(corrected_iqm_scores['Corrected']),
            corrected_cis['Corrected'][0], corrected_cis['Corrected'][1],
            label='Corrected', color=self.CORRECTED_COLOR
        )

        # 设置标题和标签
        ax.set_title(f"Sample Efficiency Curve for {task_name}", fontsize=18, fontweight='bold')
        ax.set_xlabel('Training Frames (in millions)', fontsize=16, fontweight='bold')
        ax.set_ylabel('IQM Performance Score', fontsize=16, fontweight='bold')

        # 添加图例和网格
        # ax.legend(loc='upper left', fontsize=12)
        # 绘制图例，并调整字体
        legend = ax.legend(loc='upper left', fontsize=14)  # 设置字体大小
        for label in legend.get_texts():
            label.set_fontweight('bold')  # 设置字体加粗

        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # 加粗坐标轴边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)  # 加粗左边框
        ax.spines['bottom'].set_linewidth(2)  # 加粗底边框

        # 保存图表
        save_path = os.path.join(self.output_path, "sample_efficiency", f"{task_name}_sample_efficiency.pdf")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)




def main():
    configure_plot_settings()

    data_path = "mujoco-benchmark/data-humanbench"
    output_path = "mujoco-benchmark/paintting/humanoidbench-paintting/plot_4-2/plot_4-2-2"

    processor = RLDataProcessor(data_path)
    processor.load_data()

    plotter = RLPlotter(output_path, processor)
    for task_name, data in processor.tasks_data.items():
        if not data["Standard"].size or not data["Corrected"].size:
            print(f"Skipping task {task_name} due to insufficient data.")
            continue

        print(f"Processing task: {task_name}")
        plotter.compute_aggregate_metrics(task_name, data["Standard"], data["Corrected"])
        plotter.compute_performance_profiles(task_name, data["Standard"], data["Corrected"])
        # plotter.compute_probability_of_improvement(task_name, data["Standard"], data["Corrected"])
        plotter.compute_sample_efficiency_curve(task_name, data["Standard"], data["Corrected"])

if __name__ == "__main__":
    main()