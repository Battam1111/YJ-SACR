#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

# Supported environments include HalfCheetah-v4, Hopper-v4, Swimmer-v4, Walker2d-v4, 
# Ant-v4, Humanoid-v4, Reacher-v4, InvertedPendulum-v4, InvertedDoublePendulum-v4, 
# Pusher-v4 and HumanoidStandup-v4.

def get_args():
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    
    # 指定任务环境，例如 Ant-v4
    parser.add_argument("--task", type=str, default="Humanoid-v4")
    
    # 随机种子，用于结果重现
    parser.add_argument("--seed", type=int, default=114514)
    
    # 经验回放缓冲区的大小
    parser.add_argument("--buffer-size", type=int, default=1000000)
    
    # 隐藏层的大小（神经网络结构），默认为两层，每层256个神经元
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    
    # Actor网络的学习率
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    
    # Critic网络的学习率
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    
    # 折扣因子γ，用于奖励计算
    parser.add_argument("--gamma", type=float, default=0.99)
    
    # 软更新系数τ
    parser.add_argument("--tau", type=float, default=0.005)
    
    # SAC算法中的温度参数α
    parser.add_argument("--alpha", type=float, default=0.2)
    
    # 是否自动调整α,默认不开启
    parser.add_argument("--auto-alpha", default=True, action="store_true")
    
    # α的学习率
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    
    # 初始探索的时间步数
    parser.add_argument("--start-timesteps", type=int, default=10000)
    
    # 训练的轮数，默认200
    parser.add_argument("--epoch", type=int, default=2)
    
    # 每轮中的时间步数，默认5000
    parser.add_argument("--step-per-epoch", type=int, default=2)
    
    # 每次收集的时间步数
    parser.add_argument("--step-per-collect", type=int, default=1)
    
    # 每步更新的次数
    parser.add_argument("--update-per-step", type=int, default=1)
    
    # N步回报
    parser.add_argument("--n-step", type=int, default=1)
    
    # 批量大小
    parser.add_argument("--batch-size", type=int, default=256)
    
    # 训练过程中使用的环境数量
    parser.add_argument("--training-num", type=int, default=1)
    
    # 测试过程中使用的环境数量，默认十个
    parser.add_argument("--test-num", type=int, default=1)
    
    # 日志保存路径
    parser.add_argument("--logdir", type=str, default="log")
    
    # 渲染间隔，默认不渲染？
    parser.add_argument("--render", type=float, default=0.)
    
    # 设备类型（CPU或GPU）
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 恢复训练的模型路径
    parser.add_argument("--resume-path", type=str, default=None)
    
    # 恢复训练的ID
    parser.add_argument("--resume-id", type=str, default=None)
    
    # 日志记录器类型
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    
    # WandB项目名称
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    
    # 仅观察预训练策略的表现
    parser.add_argument("--watch", default=False, action="store_true", help="watch the play of pre-trained policy only")
    
    # 算法标签
    parser.add_argument("--algo-label", type=str, default="")
    
    # 使用均匀采样
    parser.add_argument("--use-uniform-sampling", default=True, action="store_true")
    
    # 运行数（跑了多少不同种子的实验）
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs with different seeds")
    
    
    return parser.parse_args()


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
        # 测试训练
        RefineTrain=True,
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
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True, seed=args.seed)
    test_collector = Collector(policy, test_envs, seed=args.seed)
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


    # 查看训练结束时的最终表现（test环境下）
    # # 观看策略表现（确定性动作）
    # policy.eval_method = 'deterministic'
    # policy.eval()
    # test_envs.seed(args.seed)
    # test_collector.reset()
    # result_deterministic = test_collector.collect(n_episode=args.test_num, render=args.render)
    # print(f'Final reward (deterministic): {result_deterministic["rews"].mean()}, length: {result_deterministic["lens"].mean()}')

    # # 观看策略表现（均匀采样_v1）
    # policy.eval_method = 'uniform_v1'
    # policy.eval()
    # test_envs.seed(args.seed)
    # test_collector.reset()
    # result_uniform_v1 = test_collector.collect(n_episode=args.test_num, render=args.render)
    # print(f'Final reward (uniform_v1): {result_uniform_v1["rews"].mean()}, length: {result_uniform_v1["lens"].mean()}')
    
    # # 观看策略表现（均匀采样_v2）
    # policy.eval_method = 'deterministic'
    # policy.eval()
    # test_envs.seed(args.seed)
    # test_collector.reset()
    # result_uniform_v2 = test_collector.collect(n_episode=args.test_num, render=args.render)
    # print(f'Final reward (uniform_v2): {result_uniform_v2["rews"].mean()}, length: {result_uniform_v2["lens"].mean()}')

if __name__ == "__main__":
        test_sac()