import torch
import unittest
from typing import Tuple
import numpy as np

class ActionSampler:
    def __init__(self, eps=np.finfo(np.float32).eps.item()):
        self.__eps = eps

    def deterministic(self, logits: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        act = logits[0]
        squashed_action = torch.tanh(act)
        return squashed_action

    def uniform_sampling(self, logits: Tuple[torch.Tensor, torch.Tensor], num_samples: int) -> torch.Tensor:
        """基于均匀采样的方法生成动作"""
        mean, std = logits[0].clone(), logits[1].clone()
        batch_size, action_dim = mean.shape
        std = std + self.__eps  # 确保标准差不会为零或太小
        
        # 生成均匀分布的采样点
        y0 = 1 - self.__eps
        sample_range = torch.linspace(-y0, y0, num_samples).to(mean.device)
        
        # 扩展采样点的维度，以匹配批量大小和动作维度
        sample_range = sample_range.unsqueeze(0).unsqueeze(0).expand(batch_size, action_dim, num_samples)
        
        # 计算每个样本的概率密度
        log_term = (0.5 * torch.log((1 + sample_range) / (1 - sample_range) + self.__eps) - mean.unsqueeze(2)) ** 2 / (-2 * (std.unsqueeze(2) ** 2))
        probs = (1 / (1 - sample_range ** 2 + self.__eps)) * (1 / (((2 * torch.tensor(torch.pi) * std.unsqueeze(2) ** 2)) ** 0.5)) * torch.exp(log_term)
        
        # 选择每个动作中具有最高概率的样本
        best_sample_idx = torch.argmax(probs, dim=2)  # 形状为 [batch_size, action_dim]
        
        # 使用 best_sample_idx 从 sample_range 中选择最优的采样点
        best_sample = torch.gather(sample_range, 2, best_sample_idx.unsqueeze(2)).squeeze(2)
        
        return best_sample

class TestActionSampler(unittest.TestCase):
    def setUp(self):
        self.eps = np.finfo(np.float32).eps.item()
        self.sampler = ActionSampler(eps=self.eps)
        self.batch_size = 2
        self.action_dim = 2
        self.logits = (
            torch.randn(self.batch_size, self.action_dim),  # 随机均值
            torch.abs(torch.randn(self.batch_size, self.action_dim))  # 随机正标准差
        )
        self.num_samples = 2000

    def test_deterministic_prob_density(self):
        mean, std = self.logits[0], self.logits[1]
        deterministic_action = self.sampler.deterministic(self.logits)
        log_term = (0.5 * torch.log((1 + deterministic_action) / (1 - deterministic_action) + self.eps) - mean) ** 2 / (-2 * (std ** 2))
        probs = (1 / (1 - deterministic_action ** 2 + self.eps)) * (1 / (((2 * torch.tensor(torch.pi) * std ** 2)) ** 0.5)) * torch.exp(log_term)

        print(f"\nDeterministic Action Probability Densities:")
        print(probs)
        return probs

    def test_uniform_sampling_prob_density(self):
        mean, std = self.logits[0], self.logits[1]
        uniform_sample = self.sampler.uniform_sampling(self.logits, self.num_samples)
        log_term = (0.5 * torch.log((1 + uniform_sample) / (1 - uniform_sample) + self.eps) - mean) ** 2 / (-2 * (std ** 2))
        probs = (1 / (1 - uniform_sample ** 2 + self.eps)) * (1 / (((2 * torch.tensor(torch.pi) * std ** 2)) ** 0.5)) * torch.exp(log_term)

        print(f"\nUniform Sample Probability Densities:")
        print(probs)
        return probs

    def test_compare_prob_densities(self):
        probs_det = self.test_deterministic_prob_density()
        probs_uni = self.test_uniform_sampling_prob_density()

        # 比较两个概率密度
        if torch.mean(probs_det) > torch.mean(probs_uni):
            print(f"\nDeterministic Probability Densities are generally larger than Uniform Sampling Probability Densities")
        else:
            print(f"\nUniform Sampling Probability Densities are generally larger than Deterministic Probability Densities")

        print(f"\nMean Deterministic Probability Densities: {torch.mean(probs_det)}")
        print(f"Mean Uniform Sampling Probability Densities: {torch.mean(probs_uni)}")

if __name__ == '__main__':
    unittest.main()
