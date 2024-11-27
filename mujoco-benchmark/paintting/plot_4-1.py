import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义颜色常量
BASELINE_COLOR = '#498DD1'   # 蓝色，用于 Standard
ERROR_COLOR = '#ff7f0e'  # 橙色，用于 错误点
CORRECTED_COLOR = '#E15E39'     # 红色，用于 Corrected

class DistributionShift3DPlotEnhancedPeakCorrected:
    def __init__(self, mean=0.5, std=0.9):
        """
        Initialize parameters
        :param mean: Mean of the Gaussian distribution
        :param std: Standard deviation of the Gaussian distribution
        """
        self.mean = mean
        self.std = std
        
    def gaussian_distribution(self, x, y):
        """
        Calculate 2D Gaussian distribution probability density
        :param x: Action value on x-axis
        :param y: Action value on y-axis
        :return: Probability density values
        """
        return (1 / (2 * np.pi * self.std**2)) * np.exp(-((x - self.mean)**2 + (y - self.mean)**2) / (2 * self.std**2))
    
    def transformed_distribution(self, x, y):
        """
        Calculate probability density after tanh transformation
        :param x: Action value on x-axis
        :param y: Action value on y-axis
        :return: Probability density values after transformation
        """
        u_x = np.arctanh(np.clip(x, -0.999, 0.999))
        u_y = np.arctanh(np.clip(y, -0.999, 0.999))
        pdf_u = self.gaussian_distribution(u_x, u_y)
        jacobian = 1 / ((1 - x**2) * (1 - y**2))
        return pdf_u * jacobian
    
    def plot_distribution_shift(self, x_range=(-2, 2), y_range=(-2, 2), resolution=100, save_path="mujoco-benchmark/paintting/humanoidbench-paintting/plot_4-1.pdf"):
        """
        Plot the distribution shift in 3D view
        :param x_range: Range of x-axis
        :param y_range: Range of y-axis
        :param resolution: Grid resolution
        :param save_path: Path to save the plot
        """
        # Create meshgrid for x and y
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate original Gaussian and transformed distributions
        Z_original = self.gaussian_distribution(X, Y)
        X_tanh = np.clip(np.tanh(X), -1, 1)
        Y_tanh = np.clip(np.tanh(Y), -1, 1)
        Z_transformed = self.transformed_distribution(X_tanh, Y_tanh)
        
        # Create 3D plot
        fig = plt.figure(figsize=(20, 18), dpi=120)  # Larger figure, high resolution
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original Gaussian distribution
        ax.plot_surface(X, Y, Z_original, rstride=5, cstride=5, color=BASELINE_COLOR, alpha=0.5, edgecolor='none')
        
        # Plot tanh-transformed distribution
        ax.plot_surface(X_tanh, Y_tanh, Z_transformed, rstride=5, cstride=5, color=CORRECTED_COLOR, alpha=0.5, edgecolor='none')
        
        # Mark the highest point in the original Gaussian distribution
        peak_original = [self.mean, self.mean, Z_original.max()]
        ax.scatter(*peak_original, color=BASELINE_COLOR, s=120, label='Original Peak', depthshade=True)
        
        # Mark the mapped point of the original Gaussian peak on the tanh-transformed distribution
        mapped_x = np.tanh(self.mean)
        mapped_y = np.tanh(self.mean)
        mapped_z = Z_transformed[int((mapped_x + 1) * resolution / 2), int((mapped_y + 1) * resolution / 2)]
        peak_mapped = [mapped_x, mapped_y, mapped_z]
        ax.scatter(*peak_mapped, color=ERROR_COLOR, s=120, alpha=1, label='Mapped Original Peak', depthshade=True)
        
        # Calculate and mark the peak of the tanh-transformed distribution
        max_index = np.unravel_index(np.argmax(Z_transformed, axis=None), Z_transformed.shape)
        peak_tanh_max_x = X_tanh[0, max_index[1]]
        peak_tanh_max_y = Y_tanh[max_index[0], 0]
        peak_tanh_max_z = Z_transformed[max_index]
        peak_tanh_max = [peak_tanh_max_x, peak_tanh_max_y, peak_tanh_max_z]
        ax.scatter(*peak_tanh_max, color=CORRECTED_COLOR, s=120, label='Peak of Transformed Distribution', depthshade=True)
        
        # Draw vertical dashed lines for each point to show height difference
        ax.plot([peak_original[0], peak_original[0]], [peak_original[1], peak_original[1]], 
                [0, peak_original[2]], linestyle="--", color='gray', linewidth=1.5, alpha=0.7)
        ax.plot([peak_mapped[0], peak_mapped[0]], [peak_mapped[1], peak_mapped[1]], 
                [0, peak_mapped[2]], linestyle="--", color='gray', linewidth=1.5, alpha=0.7)
        ax.plot([peak_tanh_max[0], peak_tanh_max[0]], [peak_tanh_max[1], peak_tanh_max[1]], 
                [0, peak_tanh_max[2]], linestyle="--", color='gray', linewidth=1.5, alpha=0.7)
        
        # Set labels with increased font size and make them bold
        ax.set_xlabel('Action X', fontsize=18, fontweight='bold')
        ax.set_ylabel('Action Y', fontsize=18, fontweight='bold')
        ax.set_zlabel('Probability Density', fontsize=18, fontweight='bold')
        
        # Enlarge legend
        legend = ax.legend(loc='upper left', fontsize=16)
        for label in legend.get_texts():
            label.set_fontweight('bold')
        
        # Save and show plot
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)
        plt.show()

# Instantiate and plot
plotter = DistributionShift3DPlotEnhancedPeakCorrected()
plotter.plot_distribution_shift()
