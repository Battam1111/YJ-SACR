import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class CustomDistribution(nn.Module):
    def __init__(self, mean, std, num_samples=2000, y0=0.9999, validate_args=None):
        super(CustomDistribution, self).__init__()
        self._mean = mean.clone()
        self.__eps = np.finfo(np.float32).eps.item()
        self._std = std.clone() + self.__eps
        self.num_samples = num_samples
        self.y0 = y0

    def rsample(self, sample_shape=torch.Size()):
        batch_size, action_dim = self._mean.shape

        # 生成均匀分布的采样点
        sample_range = torch.linspace(-self.y0, self.y0, self.num_samples).to(self._mean.device)
        sample_range = sample_range.unsqueeze(0).unsqueeze(0).expand(batch_size, action_dim, self.num_samples)

        # 计算每个样本的对数概率密度
        log_term = (0.5 * torch.log((1 + sample_range) / (1 - sample_range) + self.__eps) - self._mean.unsqueeze(2)) ** 2 / (-2 * (self._std.unsqueeze(2) ** 2))
        log_probs = (1 / (1 - sample_range ** 2)) * (1 / (((2 * torch.tensor(torch.pi) * self._std.unsqueeze(2) ** 2)) ** 0.5)) * torch.exp(log_term)
        
        # 转换为概率
        probs = log_probs
        probs = probs / (probs.sum(dim=2, keepdim=True) + self.__eps)  # 防止除零

        # 计算CDF
        cdf = probs.cumsum(dim=2)

        # 生成均匀分布的随机数
        uniform_samples = torch.rand(batch_size, action_dim, 1, device=self._mean.device)

        # 根据CDF进行逆变换抽样
        sample_idx = (uniform_samples < cdf).int().argmax(dim=2)

        # 使用sample_idx从sample_range中选择采样点
        sampled_values = torch.gather(sample_range, 2, sample_idx.unsqueeze(2)).squeeze(2)

        # 获取所选择的采样点的概率值
        sampled_probs = torch.gather(probs, 2, sample_idx.unsqueeze(2)).squeeze(2)

        return sampled_values, sampled_probs

    def log_prob(self, value):
        atanh_value = 0.5 * torch.log((1 + value) / (1 - value) + self.__eps)
        log_term = (atanh_value - self._mean) ** 2 / (-2 * (self._std ** 2))
        
        log_probs = -torch.log(1 - value ** 2 + self.__eps) - 0.5 * torch.log(2 * torch.tensor(torch.pi, dtype=torch.float64).to(self._mean.device) * self._std ** 2) + log_term
        
        log_probs_sum = log_probs.sum(dim=1, keepdim=True)

        return log_probs_sum

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._std

def visualize_distributions(sample_range, log_probs, probs, cdf, filename=None):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(sample_range[0, 0, :].cpu().numpy())
    plt.title('Sample Range')

    plt.subplot(2, 2, 2)
    plt.plot(log_probs[0, 0, :].cpu().detach().numpy())
    plt.title('Log Probs')

    plt.subplot(2, 2, 3)
    plt.plot(probs[0, 0, :].cpu().detach().numpy())
    plt.title('Probabilities')

    plt.subplot(2, 2, 4)
    plt.plot(cdf[0, 0, :].cpu().detach().numpy())
    plt.title('CDF')

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def test_custom_distribution(save_fig=False):
    # 创建测试数据
    batch_size = 10
    action_dim = 2
    mean = torch.zeros(batch_size, action_dim)
    std = torch.ones(batch_size, action_dim) * 0.5
    num_samples = 1000
    y0 = 0.9999
    eps = np.finfo(np.float32).eps.item()

    # 创建自定义分布实例
    dist = CustomDistribution(mean, std, num_samples, y0)

    # 进行采样
    sampled_values, sampled_probs = dist.rsample()
    print("Sampled values:", sampled_values)
    print("Sampled probabilities:", sampled_probs)

    # 打印中间结果
    sample_range = torch.linspace(-y0, y0, num_samples).to(mean.device)
    sample_range = sample_range.unsqueeze(0).unsqueeze(0).expand(batch_size, action_dim, num_samples)
    log_term = (0.5 * torch.log((1 + sample_range) / (1 - sample_range) + eps) - mean.unsqueeze(2)) ** 2 / (-2 * (std.unsqueeze(2) ** 2))
    log_probs = (1 / (1 - sample_range ** 2)) * (1 / (((2 * torch.tensor(torch.pi) * std.unsqueeze(2) ** 2)) ** 0.5)) * torch.exp(log_term)
    probs = log_probs / (log_probs.sum(dim=2, keepdim=True) + eps)
    cdf = probs.cumsum(dim=2)

    # 可视化检查并保存图像
    if save_fig:
        visualize_distributions(sample_range, log_probs, probs, cdf, filename='distribution_plots.png')
    else:
        visualize_distributions(sample_range, log_probs, probs, cdf)

# 运行测试，选择是否保存图像
test_custom_distribution(save_fig=True)
