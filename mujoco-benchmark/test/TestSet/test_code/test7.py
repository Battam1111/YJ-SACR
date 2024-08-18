import unittest
import math
from numbers import Number, Real
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

__all__ = ["Normal"]


class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, num_samples=2000, y0=0.9999, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

        self.num_samples = num_samples
        self.y0 = y0


    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size(), refine=False):

        if not refine:
            shape = self._extended_shape(sample_shape)
            eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)

            return self.loc + eps * self.scale
        
        else:
            # 扩展形状以包括 sample_shape
            shape = self._extended_shape(sample_shape)
            batch_size = shape[:-1]
            action_dim = shape[-1]

            # 生成均匀分布的采样点
            sample_range = torch.linspace(-self.y0, self.y0, self.num_samples, device=self.loc.device)
            # 动态扩展 sample_range 以匹配 batch_size 和 action_dim
            sample_range = sample_range.expand(*batch_size, action_dim, self.num_samples)

            # 调整 self.loc 和 self.scale 的形状
            loc = self.loc.expand(*batch_size, action_dim)
            scale = self.scale.expand(*batch_size, action_dim)

            # 计算每个样本的对数概率密度
            log_term = (0.5 * torch.log((1 + sample_range) / (1 - sample_range)) - loc.unsqueeze(-1)) ** 2 / (-2 * (scale.unsqueeze(-1) ** 2))
            log_probs = -torch.log(1 - sample_range ** 2) - 0.5 * torch.log(2 * torch.tensor(math.pi, device=self.loc.device) * scale.unsqueeze(-1) ** 2) + log_term

            # 转换为概率
            probs = torch.exp(log_probs)
            probs = probs / (probs.sum(dim=-1, keepdim=True))

            # 计算CDF
            cdf = probs.cumsum(dim=-1)

            # 生成均匀分布的随机数
            uniform_samples = torch.rand(*batch_size, action_dim, 1, device=self.loc.device)

            # 根据CDF进行逆变换抽样
            sample_idx = (uniform_samples < cdf).int().argmax(dim=-1)

            # 使用sample_idx从sample_range中选择采样点
            sampled_values = torch.gather(sample_range, -1, sample_idx.unsqueeze(-1)).squeeze(-1)

            # tanh的逆变换
            sample = torch.atanh(sampled_values)

            return sample

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (
            1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2))
        )

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)


class TestNormalDistribution(unittest.TestCase):
    def setUp(self):
        self.loc = torch.tensor([0.0])
        self.scale = torch.tensor([1.0])
        self.dist = Normal(self.loc, self.scale)
        self.sample_shape = torch.Size([1, 5])
    
    def test_mean(self):
        self.assertTrue(torch.equal(self.dist.mean, self.loc))
    
    def test_mode(self):
        self.assertTrue(torch.equal(self.dist.mode, self.loc))
    
    def test_stddev(self):
        self.assertTrue(torch.equal(self.dist.stddev, self.scale))
    
    def test_variance(self):
        self.assertTrue(torch.equal(self.dist.variance, self.scale.pow(2)))
    
    def test_sample(self):
        sample = self.dist.sample(self.sample_shape)
        self.assertEqual(sample.shape, self.sample_shape + self.loc.shape)
    
    def test_rsample(self):
        rsample = self.dist.rsample(self.sample_shape)
        self.assertEqual(rsample.shape, self.sample_shape + self.loc.shape)
    
    def test_rsample_with_refine(self):
        rsample = self.dist.rsample(self.sample_shape, refine=True)
        self.assertEqual(rsample.shape, self.sample_shape + self.loc.shape)
    
    def test_log_prob(self):
        value = torch.tensor([0.0])
        log_prob = self.dist.log_prob(value)
        expected_log_prob = -0.5 * math.log(2 * math.pi) - 0.5 * value.pow(2)
        self.assertAlmostEqual(log_prob.item(), expected_log_prob.item(), places=5)
    
    def test_cdf(self):
        value = torch.tensor([0.0])
        cdf_value = self.dist.cdf(value)
        expected_cdf_value = torch.tensor([0.5])
        self.assertAlmostEqual(cdf_value.item(), expected_cdf_value.item(), places=5)
    
    def test_icdf(self):
        value = torch.tensor([0.5])
        icdf_value = self.dist.icdf(value)
        expected_icdf_value = self.loc
        self.assertAlmostEqual(icdf_value.item(), expected_icdf_value.item(), places=5)
    
    def test_entropy(self):
        entropy = self.dist.entropy()
        expected_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + math.log(self.scale.item())
        self.assertAlmostEqual(entropy.item(), expected_entropy, places=5)
    
    def test_natural_params(self):
        nat_params = self.dist._natural_params
        expected_nat_params = (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())
        self.assertTrue(torch.equal(nat_params[0], expected_nat_params[0]))
        self.assertTrue(torch.equal(nat_params[1], expected_nat_params[1]))
    
    def test_log_normalizer(self):
        x, y = torch.tensor([1.0]), torch.tensor([-0.5])
        log_norm = self.dist._log_normalizer(x, y)
        expected_log_norm = -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)
        self.assertAlmostEqual(log_norm.item(), expected_log_norm.item(), places=5)

if __name__ == '__main__':
    unittest.main()
