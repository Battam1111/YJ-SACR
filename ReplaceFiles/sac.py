from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import csv
import copy
import sys
import os
from torch.distributions import Independent, Normal
from torch.distributions import Distribution

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy


class SACPolicy(DDPGPolicy):
    """软演员-评论家算法（Soft Actor-Critic）的实现。参考arXiv:1812.05905。

    :param torch.nn.Module actor: 扮演者（演员）网络，遵循
        :class:`~tianshou.policy.BasePolicy`中的规则。 (s -> logits)
    :param torch.optim.Optimizer actor_optim: 演员网络的优化器。
    :param torch.nn.Module critic1: 第一个评论家网络。 (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: 第一个评论家网络的优化器。
    :param torch.nn.Module critic2: 第二个评论家网络。 (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: 第二个评论家网络的优化器。
    :param float tau: 目标网络的软更新参数。默认值为0.005。
    :param float gamma: 折扣因子，范围在[0, 1]之间。默认值为0.99。
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: 熵正则化系数。默认值为0.2。
        如果提供的是元组（target_entropy, log_alpha, alpha_optim），那么alpha会被自动调整。
    :param bool reward_normalization: 是否将奖励标准化为Normal(0, 1)。默认值为False。
    :param BaseNoise exploration_noise: 为探索添加的噪声。默认值为None。解决难以探索的问题时有用。
    :param bool deterministic_eval: 是否使用确定性动作（高斯策略的均值）而不是通过策略采样的随机动作。默认值为True。
    :param bool action_scaling: 是否将动作从范围[-1, 1]映射到[action_spaces.low, action_spaces.high]。默认值为True。
    :param str action_bound_method: 将动作限制在范围[-1, 1]的方法，可以是"clip"（简单裁剪动作）或为空字符串（无边界）。默认值为"clip"。
    :param Optional[gym.Space] action_space: 环境的动作空间，如果使用“action_scaling”或“action_bound_method”选项，必须指定。默认值为None。
    :param lr_scheduler: 一个学习率调度器，在每次policy.update()中调整优化器的学习率。默认值为None（无学习率调度器）。
    :param bool use_uniform_sampling: 是否使用均匀采样方法。默认值为False。

    .. seealso::

        请参阅 :class:`~tianshou.policy.BasePolicy` 获取更详细的解释。
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        use_uniform_sampling: bool = False,
        num_samples: int = 2000,
        eval_method: str = "deterministic",
        RefineTrain_Num: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None, None, None, None, tau, gamma, exploration_noise,
            reward_normalization, estimation_step, **kwargs
        )
        # 初始化演员和评论家网络及其优化器
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        # 初始化alpha参数
        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()  # 浮点数最小值

        # 新增参数
        self.use_uniform_sampling = use_uniform_sampling
        self.num_samples = num_samples
        self.eval_method = eval_method
        self.RefineTrain_Num = RefineTrain_Num
        self.epoch = 0

    def train(self, mode: bool = True) -> "SACPolicy":
        """设置训练模式"""
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        """同步权重到目标网络"""
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """前向传播"""
        obs = batch[input]
        logits, hidden = self.actor(obs, state=state, info=batch.info)

        assert isinstance(logits, tuple)

        RefineTrain = (self.RefineTrain_Num <= self.epoch) and (self.RefineTrain_Num != -1) and self.training

        dist_P = Independent(Normal(*logits), 1)

        if self.eval_method == 'deterministic' and not self.training:
            act = logits[0]  # 使用确定性动作
            squashed_action = torch.tanh(act)

            log_prob = dist_P.log_prob(act).unsqueeze(-1)
            log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        elif (self.eval_method == 'uniform_v1' and not self.training):
            squashed_action = self.uniform_sampling(logits, self.num_samples)  # 使用均匀采样
            atanh_act = torch.atanh(squashed_action)

            log_prob = dist_P.log_prob(atanh_act).unsqueeze(-1)
            log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        else:
            act = dist_P.rsample(refine=RefineTrain) # 采样动作

            log_prob = dist_P.log_prob(act).unsqueeze(-1)

            squashed_action = torch.tanh(act)
            log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)
        

        return Batch(
            logits=logits,
            act=squashed_action,
            state=hidden,
            dist=dist_P,
            log_prob=log_prob
        )

    def uniform_sampling(self, logits: Tuple[torch.Tensor, torch.Tensor], num_samples: int) -> torch.Tensor:
        """基于均匀采样的方法生成动作"""
        mean, std = logits[0].clone(), logits[1].clone()
        batch_size, action_dim = mean.shape
        std = std + self.__eps  # 确保标准差不会为零或太小
        

        # 生成均匀分布的采样点
        y0 = 0.999
        sample_range = torch.linspace(-y0, y0, num_samples).to(mean.device)
        
        # 扩展采样点的维度，以匹配批量大小和动作维度
        sample_range = sample_range.unsqueeze(0).unsqueeze(0).expand(batch_size, action_dim, num_samples)
        

        # 计算每个样本的概率密度（原版）
        log_term = (0.5 * torch.log((1 + sample_range) / (1 - sample_range) + self.__eps) - mean.unsqueeze(2)) ** 2 / (-2 * (std.unsqueeze(2) ** 2))
        log_probs = (1 / (1 - sample_range ** 2 + self.__eps)) * (1 / (((2 * torch.tensor(torch.pi) * std.unsqueeze(2) ** 2)) ** 0.5)) * torch.exp(log_term)
        
        
        # 选择每个动作中具有最高概率的样本
        best_sample_idx = torch.argmax(log_probs, dim=2)  # 形状为 [batch_size, action_dim]
        
        
        # 使用 best_sample_idx 从 sample_range 中选择最优的采样点
        best_sample = torch.gather(sample_range, 2, best_sample_idx.unsqueeze(2)).squeeze(2)
        
        return best_sample

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """计算目标Q值"""
        batch = buffer[indices]  # 获取对应批次数据
        obs_next_result = self(batch, input="obs_next")
        act_ = obs_next_result.act
        target_q = torch.min(
            self.critic1_old(batch.obs_next, act_),
            self.critic2_old(batch.obs_next, act_),
        ) - self._alpha * obs_next_result.log_prob
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """更新策略"""
        # 更新评论家1和评论家2网络
        td1, critic1_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optim
        )
        td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # 更新优先经验回放权重

        # 更新演员网络
        obs_result = self(batch)
        act = obs_result.act
        current_q1a = self.critic1(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        actor_loss = (
            self._alpha * obs_result.log_prob.flatten() -
            torch.min(current_q1a, current_q2a)
        ).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 如果启用了自动调整alpha
        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        # 同步目标网络权重
        self.sync_weight()

        # 返回损失结果
        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result