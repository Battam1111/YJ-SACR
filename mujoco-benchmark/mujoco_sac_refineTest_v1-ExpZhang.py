#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
# torch.set_num_threads(1)

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
    parser.add_argument("--seed", type=int, default=0)
    
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
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    
    # α的学习率
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    
    # 初始探索的时间步数
    parser.add_argument("--start-timesteps", type=int, default=10000)
    
    # 训练的轮数，默认200
    parser.add_argument("--epoch", type=int, default=200)
    
    # 每轮中的时间步数，默认5000
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    
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
    
    # 测试过程中使用的环境数量，默认是10
    parser.add_argument("--test-num", type=int, default=10)
    
    # 日志保存路径
    parser.add_argument("--logdir", type=str, default="log")
    
    # 渲染间隔
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
        # wandb更方便
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    
    # WandB项目名称
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    
    # 仅观察预训练策略的表现
    parser.add_argument("--watch", default=False, action="store_true", help="watch the play of pre-trained policy only")
    
    # 算法标签
    parser.add_argument("--algo-label", type=str, default="")
    
    # 切换训练方法的开始轮数
    parser.add_argument("--RefineTrain_Num", type=int, default=-1)
    
    return parser.parse_args()


def additional_training(policy, replay_buffer, steps_list, test_collector, args):
    """
    使用Replay Buffer对policy net进行额外训练，并评估效果（仅训练actor）。
    
    :param policy: SAC策略
    :param replay_buffer: 已存储的Replay Buffer
    :param steps_list: 训练步数列表 [100, 1000, 10000]
    :param test_collector: 测试数据收集器
    :param args: 实验参数
    :return: 训练后的性能结果
    """
    results = {}
    for steps in steps_list:
        print(f"\n--- 开始额外训练（仅训练 Actor）: {steps}步 ---")
        policy.train()

        for step in range(steps):
            # 从 Replay Buffer 采样并处理数据
            batch, indices = replay_buffer.sample(args.batch_size)
            batch = policy.process_fn(batch, replay_buffer, indices)

            # 调用更新方法
            losses = policy.learn_actor(batch=batch)

            if step % 100 == 0:
                print(f"训练步数: {step}, Losses: {losses}")
        
        # 测试训练后的性能
        policy.eval_method = 'deterministic'
        policy.eval()
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        avg_reward = result["rews"].mean()
        avg_length = result["lens"].mean()
        print(f"额外训练 {steps} 步后平均奖励: {avg_reward}, 平均长度: {avg_length}")
        
        results[steps] = {
            "avg_reward": avg_reward,
            "avg_length": avg_length
        }
    
    return results





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
        RefineTrain_Num=args.RefineTrain_Num,
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
    test_collector = Collector(policy, test_envs, task=args.task, seed=args.seed)
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
        # 正常训练
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
            verbose=False,
        )
        pprint.pprint(result)
        
        # 保存正常训练后的模型
        final_policy_path = os.path.join(log_path, "final_policy.pth")
        torch.save(policy.state_dict(), final_policy_path)
        print(f"训练完成，模型已保存至: {final_policy_path}")


        # 执行额外训练
        steps_list = [100, 1000, 10000]
        extra_results = additional_training(policy, buffer, steps_list, test_collector, args)

        # 保存所有结果到一个日志文件
        combined_log_file = os.path.join(log_path, "training_results.txt")
        with open(combined_log_file, "w") as f:
            # 保存正常训练结果
            f.write("=== 正常训练结果 ===\n")
            for key, value in result.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # 保存额外训练结果
            f.write("=== 额外训练结果 ===\n")
            for steps, extra_result in extra_results.items():
                f.write(f"训练步数: {steps}\n")
                f.write(f"平均奖励: {extra_result['avg_reward']}\n")
                f.write(f"平均长度: {extra_result['avg_length']}\n")
                f.write("\n")
        print(f"训练结果已保存至: {combined_log_file}")

    
    

if __name__ == "__main__":
    args = get_args()
    # 随机跑五个种子
    # seeds = [0,1,2,3,4]
    # seeds = [5,6,7,8,9]
    seeds = [0]
    tasks = ["HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4", 
             "Ant-v4", "Humanoid-v4", "Reacher-v4", 
             "InvertedPendulum-v4", "InvertedDoublePendulum-v4", 
             "Pusher-v4", "HumanoidStandup-v4"]
    
    # 可选，将所有任务跑一遍
    # for task in tasks:
    #     args.task = task
        
    #     for seed in seeds:
    #         args.seed = seed
    #         test_sac(args)

    # 指定某个任务进行实验
    args.task = 'Ant-v4'

    # 指定是否自动调alpha
    args.auto_alpha = False

    # 指定alpha0.02、0.1、0.5、0.2
    # args.alpha = 0.05

    # args.epoch = 10000
    # args.step_per_epoch = 20

    # 是否恢复训练
    # args.resume_path = "log/Humanoid-v4/sac/0/240716-185405/policy.pth"

    # 是否开启Refine训练模式(0 OR -1)
    args.RefineTrain_Num = -1

    for seed in seeds:
        args.seed = seed
        test_sac(args)
