#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, Batch
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union
from tianshou.exploration import BaseNoise


def get_args():
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    # parser.add_argument("--epoch", type=int, default=200)
    # 测试方法是否有效
    parser.add_argument("--epoch", type=int, default=1)
    # parser.add_argument("--step-per-epoch", type=int, default=5000)
    # 测试方法是否有效
    parser.add_argument("--step-per-epoch", type=int, default=1)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",  # 实际运行时使用wandb
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--algo-label", type=str, default="")
    return parser.parse_args()

# 基于均匀采样的策略评估类
class SampleBasedSACPolicy(SACPolicy):
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
        num_samples: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
            tau, gamma, alpha, reward_normalization, estimation_step,
            exploration_noise, deterministic_eval, **kwargs
        )
        self.num_samples = num_samples

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """前向传播，基于均匀采样的方法"""
        obs = batch[input]
        logits, hidden = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        means, stds = logits
        
        # 生成均匀分布的采样点
        samples = np.linspace(-1 + 1e-6, 1 - 1e-6, self.num_samples)  # 确保动作在 (-1 + ε, 1 - ε) 范围内
        samples = torch.tensor(samples, device=means.device, dtype=means.dtype)
        
        # 调整 samples 形状以匹配 means 和 stds
        samples = samples.view(-1, 1, 1)  # (num_samples, 1, 1)
        means = means.view(1, *means.shape)  # (1, batch_size, action_dim)
        stds = stds.view(1, *stds.shape)  # (1, batch_size, action_dim)

        # 计算每个样本的概率
        log_samples = torch.log((1 + samples) / (1 - samples))  # (num_samples, 1, 1)
        probabilities = torch.exp(-0.5 * ((log_samples - means) / stds) ** 2) / (stds * np.sqrt(2 * np.pi))  # (num_samples, batch_size, action_dim)
        probabilities = probabilities.prod(dim=-1)  # (num_samples, batch_size)

        # 额外的修正因子
        adjustment = 1 / torch.clamp(1 - samples**2, min=1e-6)  # (num_samples, 1, 1)
        adjustment = adjustment.squeeze(-1).squeeze(-1)  # (num_samples,)

        # 调整 probabilities 形状以匹配 adjustment
        probabilities = probabilities * adjustment[:, None]  # (num_samples, batch_size)

        # 找到具有最高概率的样本
        best_sample_idx = torch.argmax(probabilities, dim=0)  # (batch_size,)
        best_sample = samples[best_sample_idx, :, :]  # 获取最佳样本，形状为 (batch_size, 1, 1)

        # 调整 best_sample 形状
        best_sample = best_sample.squeeze(-1).squeeze(-1)  # (batch_size,)

        # 返回最优动作
        return Batch(act=best_sample)



def test_sac(args=get_args()):
    # 创建环境
    env, train_envs, test_envs = make_mujoco_env(
        args.task, args.seed, args.training_num, args.test_num, obs_norm=False
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 模型
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # 自动调整alpha值
    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # 创建SAC策略
    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # 基于采样的方法来评估最优策略
    sample_based_policy = SampleBasedSACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        tau=args.tau, gamma=args.gamma, alpha=args.alpha, num_samples=2000,
        deterministic_eval=False, action_space=env.action_space
    )

    # 加载之前训练的策略
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # 创建数据收集器
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)  # 传统方法的测试收集器
    sample_test_collector = Collector(sample_based_policy, test_envs)  # 基于采样方法的测试收集器
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # 日志设置
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "sac"
    args.algo_label = "" if args.algo_label == "" else " " + args.algo_label
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now + args.algo_label)
    log_path = os.path.join(args.logdir, log_name)

    # 日志记录器
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        # 保存最优策略
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # 训练器
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        )
        pprint.pprint(result)

    # 观看传统方法策略表现
    print("Testing traditional method...")
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Traditional method - Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

    # 观看基于采样方法策略表现
    print("Testing sample-based method...")
    sample_based_policy.actor.eval()
    sample_test_collector.reset()
    result = sample_test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Sample-based method - Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    test_sac()
