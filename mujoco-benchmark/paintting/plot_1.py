import matplotlib.pyplot as plt
import numpy as np

# 定义颜色常量
BASELINE_COLOR = '#498DD1'   # 蓝色，用于 Standard
ERROR_COLOR = '#ff7f0e'  # 橙色，用于 错误点
CORRECTED_COLOR = '#E15E39'     # 红色，用于 Corrected

class DistributionShiftPlotSave:
    def __init__(self, mean=0.5, std=0.5):
        """
        初始化类的参数
        :param mean: 原始高斯分布的均值
        :param std: 原始高斯分布的标准差
        """
        self.mean = mean  # 高斯分布均值
        self.std = std    # 高斯分布标准差
        
    def gaussian_distribution(self, x):
        """
        计算原始高斯分布的概率密度函数
        :param x: 输入的动作值
        :return: 对应的概率密度值
        """
        return (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
    
    def transformed_distribution(self, x):
        """
        计算经过tanh变换后的分布的概率密度函数
        :param x: 输入的动作值
        :return: 变换后分布的概率密度
        """
        u = np.arctanh(np.clip(x, -0.999, 0.999))  # 反tanh操作
        pdf_u = self.gaussian_distribution(u)      # 计算原始分布在反tanh点的值
        jacobian = 1 / (1 - x**2)                  # 计算tanh变换的雅可比行列式
        return pdf_u * jacobian                    # 得到变换后的密度
    
    def plot_distributions(self, x_range=(-3, 3), num_points=1000, save_path="mujoco-benchmark/paintting/humanoidbench-paintting/plot_1.pdf"):
        """
        绘制原始高斯分布和经过tanh变换后的分布，并保存为pdf格式
        :param x_range: 横轴的范围
        :param num_points: 绘制的点数
        :param save_path: 图像保存的路径，默认保存为pdf
        """
        x = np.linspace(x_range[0], x_range[1], num_points)  # 生成动作值的范围
        
        # 计算原始分布和变换后分布的概率密度
        original_pdf = self.gaussian_distribution(x)  # 原始高斯分布
        transformed_pdf = self.transformed_distribution(np.tanh(x))  # tanh变换后的分布
        
        # 找到原始高斯分布中概率密度最高点（均值点）及其映射
        max_density_point = self.mean
        max_density_tanh = np.tanh(max_density_point)  # 映射到tanh变换后的位置
        
        # 找到tanh变换后分布的最高点
        max_transformed_index = np.argmax(transformed_pdf)
        max_transformed_x = np.tanh(x[max_transformed_index])  # 获取最高点的x值
        max_transformed_y = transformed_pdf[max_transformed_index]  # 获取最高点的y值
        
        # 开始绘图，增加画布尺寸
        plt.figure(figsize=(14, 8))
        
        # 绘制原始高斯分布
        plt.plot(x, original_pdf, label="Original Gaussian Distribution", color=BASELINE_COLOR, linestyle='--', linewidth=3)
        
        # 绘制经过tanh变换后的分布
        plt.plot(np.tanh(x), transformed_pdf, label="Transformed Distribution", color=CORRECTED_COLOR, linewidth=3)
        
        # 标注原始高斯分布中概率密度最高点
        plt.plot(max_density_point, self.gaussian_distribution(max_density_point), 'o', color=BASELINE_COLOR, 
                 markersize=12, label="Original Peak")

        # 标注映射后的位置
        plt.plot(max_density_tanh, self.transformed_distribution(max_density_tanh), 'o', color=ERROR_COLOR, 
                 markersize=12, label="Mapped Original Peak")
        
        # 标注tanh变换后分布的最高点
        plt.plot(max_transformed_x, max_transformed_y, 'o', color=CORRECTED_COLOR, markersize=12,
                 label="Peak of Transformed Distribution")
        
        # 添加注释和图例
        plt.xlabel("Action Values", fontsize=18, fontweight='bold')
        plt.ylabel("Probability Density", fontsize=18, fontweight='bold')
        
        # 绘制图例，并调整字体
        legend = plt.legend(fontsize=16,loc='upper left')  # 设置字体大小
        for label in legend.get_texts():
            label.set_fontweight('bold')  # 设置字体加粗

        # 恢复为标准网格背景并加粗网格线
        plt.grid(True, linewidth=0.75)
        
        # 保存图像
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)
        
        # 显示图表
        plt.show()

# 创建类的实例并绘制最终图像，并保存为pdf格式
final_plotter_save = DistributionShiftPlotSave()
final_plotter_save.plot_distributions()
