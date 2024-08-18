import torch
import numpy as np
import unittest
from torch.distributions import Distribution

class CustomDistribution(Distribution):
    def __init__(self, mean, std, num_samples=3, y0=0.9999, validate_args=None):
        self._mean = mean.clone()
        self.__eps = np.finfo(np.float32).eps.item()  # 浮点数最小值
        self._std = std.clone() + self.__eps  # 防止标准差太小
        self.num_samples = num_samples
        self.y0 = y0
        batch_shape = mean.size()
        super(CustomDistribution, self).__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        batch_size, action_dim = self._mean.shape

        # 生成均匀分布的采样点
        sample_range = torch.linspace(-self.y0, self.y0, self.num_samples).to(self._mean.device)
        sample_range = sample_range.unsqueeze(0).unsqueeze(0).expand(batch_size, action_dim, self.num_samples)

        # 计算每个样本的对数概率密度
        log_term = (0.5 * torch.log((1 + sample_range) / (1 - sample_range) + self.__eps) - self._mean.unsqueeze(2)) ** 2 / (-2 * (self._std.unsqueeze(2) ** 2))
        log_probs = -torch.log(1 - sample_range ** 2 + self.__eps) - 0.5 * torch.log(2 * torch.tensor(torch.pi) * self._std.unsqueeze(2) ** 2) + log_term

        # 转换为概率
        probs = torch.exp(log_probs)
        probs = probs / probs.sum(dim=2, keepdim=True)

        # 计算CDF
        cdf = probs.cumsum(dim=2)

        # 生成均匀分布的随机数
        uniform_samples = torch.rand(batch_size, action_dim, 1, device=self._mean.device)

        # 根据CDF进行逆变换抽样
        sample_idx = (uniform_samples < cdf).int().argmax(dim=2)

        # 使用sample_idx从sample_range中选择采样点
        sampled_values = torch.gather(sample_range, 2, sample_idx.unsqueeze(2)).squeeze(2)

        return sampled_values


    def log_prob(self, value):

        # 计算 log_term
        log_term = (0.5 * torch.log((1 + value) / (1 - value) +  self.__eps) - self._mean) ** 2 / (-2 * (self._std ** 2))
        
        # 计算 log_probs
        log_probs = -torch.log(1 - value ** 2 + self.__eps) - 0.5 * torch.log(2 * torch.tensor(torch.pi) * self._std ** 2) + log_term
        
        # 对每个动作维度的 log_prob 求和，返回形状为 [批次数, 1] 的张量
        log_probs_sum = log_probs.sum(dim=1, keepdim=True)

        return log_probs_sum


class TestCustomDistribution(unittest.TestCase):
    def setUp(self):
        self.mean = torch.tensor([[0.0, 0.0], [0.5, -0.5]], requires_grad=False)
        self.std = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=False)
        self.dist = CustomDistribution(self.mean, self.std)

    def test_rsample_shape(self):
        samples = self.dist.rsample()
        self.assertEqual(samples.shape, self.mean.shape)

    def test_rsample_values(self):
        samples = self.dist.rsample()
        self.assertTrue(torch.all(samples >= -self.dist.y0))
        self.assertTrue(torch.all(samples <= self.dist.y0))

    def test_log_prob_shape(self):
        samples = self.dist.rsample()
        log_probs = self.dist.log_prob(samples)
        self.assertEqual(log_probs.shape, (self.mean.shape[0], 1))

    def test_log_prob_values(self):
        samples = self.dist.rsample()
        log_probs = self.dist.log_prob(samples)
        self.assertTrue(torch.all(log_probs <= 0))  # log probability should be <= 0

    def test_deterministic_output(self):
        torch.manual_seed(0)
        samples1 = self.dist.rsample()
        torch.manual_seed(0)
        samples2 = self.dist.rsample()
        self.assertTrue(torch.allclose(samples1, samples2))

if __name__ == "__main__":
    unittest.main()
