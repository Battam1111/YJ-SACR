# Re-defining necessary imports and class as the environment has been reset

import numpy as np
import matplotlib.pyplot as plt

# 定义颜色常量
BASELINE_COLOR = '#498DD1'   # 蓝色，用于 Standard
ERROR_COLOR = '#ff7f0e'  # 橙色，用于 错误点
CORRECTED_COLOR = '#E15E39'     # 红色，用于 Corrected

# 定义绘制多组不同均值和标准差下分布图的类
class DistributionShiftPlotMultiMeanStdAllLegend:
    def __init__(self, means, stds):
        """
        初始化类的参数
        :param means: 高斯分布均值的列表
        :param stds: 高斯分布标准差的列表
        """
        self.means = means  # 各组高斯分布的均值
        self.stds = stds    # 各组高斯分布的标准差
        
    def gaussian_distribution(self, x, mean, std):
        """
        计算给定均值和标准差下的高斯分布概率密度函数
        :param x: 输入的动作值
        :param mean: 高斯分布的均值
        :param std: 高斯分布的标准差
        :return: 对应的概率密度值
        """
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    def transformed_distribution(self, x, mean, std):
        """
        计算给定均值和标准差下经过tanh变换后的分布概率密度函数
        :param x: 输入的动作值
        :param mean: 高斯分布的均值
        :param std: 高斯分布的标准差
        :return: 变换后分布的概率密度
        """
        u = np.arctanh(np.clip(x, -0.999, 0.999))  # 反tanh操作
        pdf_u = self.gaussian_distribution(u, mean, std)  # 在反tanh点计算原始高斯分布
        jacobian = 1 / (1 - x**2)  # 计算tanh变换的雅可比行列式
        return pdf_u * jacobian    # 得到变换后的密度
    
    def plot_distributions(self, x_range=(-3, 3), num_points=1000, save_path="mujoco-benchmark/paintting/humanoidbench-paintting/plot_3-1.pdf"):
        """
        绘制不同均值和标准差下的原始高斯分布和经过tanh变换后的分布，并标注各自的均值和标准差
        :param x_range: 横轴的范围
        :param num_points: 绘制的点数
        :param save_path: 图像保存的路径，默认保存为pdf
        """
        x = np.linspace(x_range[0], x_range[1], num_points)  # 生成动作值的范围
        tanh_x = np.tanh(x)  # tanh映射的x值范围
        
        # 开始绘图，增加画布尺寸
        plt.figure(figsize=(14, 8))
        
        # 遍历每组均值和标准差，绘制对应的曲线并在图例中标注
        for mean, std, alpha in zip(self.means, self.stds, np.linspace(0.3, 1, len(self.means))):
            # 计算原始高斯分布和变换后分布的概率密度
            original_pdf = self.gaussian_distribution(x, mean, std)  # 原始高斯分布
            transformed_pdf = self.transformed_distribution(tanh_x, mean, std)  # tanh变换后的分布
            
            # 绘制原始高斯分布，透明度调整为alpha
            plt.plot(x, original_pdf, color=BASELINE_COLOR, linestyle='--', linewidth=2, alpha=alpha,
                     label=f"Original Gaussian (mean={mean}, std={std})")
            
            # 绘制经过tanh变换后的分布，透明度调整为alpha
            plt.plot(tanh_x, transformed_pdf, color=CORRECTED_COLOR, linewidth=2, alpha=alpha,
                     label=f"Transformed Distribution (mean={mean}, std={std})")
        
        # 设置标签和图例
        plt.xlabel("Action Values", fontsize=18, fontweight='bold')
        plt.ylabel("Probability Density", fontsize=18, fontweight='bold')
        
        # 在右上角显示完整的图例，包含每组的均值和标准差信息
        # plt.legend(loc='upper right', frameon=False, fontsize=10)

        # 绘制图例，并调整字体
        legend = plt.legend(fontsize=12, frameon=False, loc='upper left')  # 设置字体大小
        for label in legend.get_texts():
            label.set_fontweight('bold')  # 设置字体加粗

        # 设置标准网格背景
        plt.grid(True, linewidth=0.75)
        
        # 保存图像
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)
        
        # 显示图表
        plt.show()

# 创建类的实例并绘制带有所有组信息的图像
multi_mean_std_all_legend_plotter = DistributionShiftPlotMultiMeanStdAllLegend(means=[-1.0, 0, 1.0], stds=[0.5, 0.5, 0.5])
multi_mean_std_all_legend_plotter.plot_distributions()

