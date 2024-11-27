import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from typing import Tuple  # 导入类型提示Tuple

# 定义一个动作采样器类
class ActionSampler:
    def __init__(self, eps=np.finfo(np.float32).eps.item()):
        # 初始化方法，设置一个极小值 eps，用于避免除零或标准差过小的问题
        self.eps = eps

    def deterministic(self, logits: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # 定义一个确定性方法，从 logits 中返回确定性动作
        act = logits[0]  # 获取第一个张量，代表动作均值
        squashed_action = torch.tanh(act)  # 使用tanh函数压缩动作，使其范围在[-1, 1]之间
        return squashed_action  # 返回压缩后的动作

    def uniform_sampling(self, logits: Tuple[torch.Tensor, torch.Tensor], num_samples: int) -> torch.Tensor:
        """基于均匀采样的方法生成动作"""
        mean, std = logits[0].clone(), logits[1].clone()  # 克隆 logits 中的均值和标准差
        batch_size, action_dim = mean.shape  # 获取批量大小和动作维度
        std = std + self.eps  # 确保标准差不会为零或太小
        
        # 生成均匀分布的采样点
        y0 = 1 - self.eps  # 设置采样范围的上限
        sample_range = torch.linspace(-y0, y0, num_samples).to(mean.device)  # 在[-1, 1]之间均匀生成采样点
        
        # 扩展采样点的维度，以匹配批量大小和动作维度
        sample_range = sample_range.unsqueeze(0).unsqueeze(0).expand(batch_size, action_dim, num_samples)
        
        # 计算每个样本的概率密度
        log_term = (0.5 * torch.log((1 + sample_range) / (1 - sample_range) + self.eps) - mean.unsqueeze(2)) ** 2 / (-2 * (std.unsqueeze(2) ** 2))
        # 概率密度计算公式：基于对数正态分布
        probs = (1 / (1 - sample_range ** 2 + self.eps)) * (1 / (((2 * torch.tensor(torch.pi) * std.unsqueeze(2) ** 2)) ** 0.5)) * torch.exp(log_term)
        
        # 选择每个动作中具有最高概率的样本
        best_sample_idx = torch.argmax(probs, dim=2)  # 找到概率密度最大的索引，形状为 [batch_size, action_dim]
        
        # 使用 best_sample_idx 从 sample_range 中选择最优的采样点
        best_sample = torch.gather(sample_range, 2, best_sample_idx.unsqueeze(2)).squeeze(2)
        # 从采样点中选取概率最大的那个值作为最优动作
        
        return best_sample  # 返回最优采样点

def compute_prob_density(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float) -> torch.Tensor:
    """计算给定动作的概率密度"""

    if action.shape != mean.shape or action.shape != std.shape:
        action = action.unsqueeze(0)

    log_term = (0.5 * torch.log((1 + action) / (1 - action) + eps) - mean) ** 2 / (-2 * (std ** 2))
    probs = (1 / (1 - action ** 2 + eps)) * (1 / (((2 * torch.tensor(torch.pi) * std ** 2)) ** 0.5)) * torch.exp(log_term)
    return probs

# 生成样本数据
def generate_samples(sampler: ActionSampler, logits: Tuple[torch.Tensor, torch.Tensor], num_samples: int):
    # 使用确定性和均匀采样方法生成样本
    deterministic_sample = sampler.deterministic(logits)
    uniform_sample = sampler.uniform_sampling(logits, num_samples)
    
    return deterministic_sample, uniform_sample

def plot_probability_density(mean: torch.Tensor, std: torch.Tensor, deterministic_sample: torch.Tensor, uniform_sample: torch.Tensor, eps: float, num_samples: int):
    """绘制概率密度函数图像并标注采样点"""
    x = torch.linspace(-1 + eps, 1 - eps, num_samples)  # 生成[-1, 1]范围内的点
    y = compute_prob_density(x, mean, std, eps)  # 计算这些点对应的概率密度
    
    if y.shape != x.shape:
        y = y.squeeze(0)

    plt.figure(figsize=(10, 6))  # 创建图像，设置大小
    plt.plot(x.numpy(), y.numpy(), label='Probability Density')  # 绘制概率密度函数
    
    # 绘制确定性采样点的位置
    for i in range(deterministic_sample.shape[0]):
        plt.axvline(x=deterministic_sample[i].item(), color='r', linestyle='--', label='Deterministic Sample' if i == 0 else "")
        # 绘制均匀采样点的位置
        plt.axvline(x=uniform_sample[i].item(), color='g', linestyle='-.', label='Refine Sample' if i == 0 else "")
    
    plt.title('Probability Density Function with Sample Points')  # 设置图像标题
    plt.xlabel('Action')  # 设置x轴标签
    plt.ylabel('Density')  # 设置y轴标签
    plt.legend()  # 显示图例W
    plt.grid(True)  # 显示网格
    plt.savefig((f'mujoco-benchmark/paintting/motivation-paintting/Probability-Density.pdf'), dpi=600)  # 保存为PDF格式
    plt.close()

# 主程序
if __name__ == '__main__':
    # 创建动作采样器实例
    sampler = ActionSampler()
    batch_size = 1
    action_dim = 1
    mean = torch.randn(batch_size, action_dim)  # 随机均值
    std = torch.abs(torch.randn(batch_size, action_dim))  # 随机正标准差
    # 将均值和标准差作为logits传入
    logits = (
            mean,
            std
        )
    
    # 设定采样数量
    num_samples = 2000
    
    # 生成确定性和均匀采样样本
    det_sample, uni_sample = generate_samples(sampler, logits, num_samples)
    # 绘制概率密度函数并标注采样点
    plot_probability_density(mean, std, det_sample, uni_sample, sampler.eps, num_samples)
