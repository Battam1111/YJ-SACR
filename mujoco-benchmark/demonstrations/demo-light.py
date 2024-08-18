import gymnasium as gym
import torch
import numpy as np
import os
from tianshou.data import Batch
from gymnasium.wrappers.record_video import RecordVideo

from ddpg_policy import get_ddpg_policy
from td3_policy import get_td3_policy
from sac_policy import get_sac_policy
from redq_policy import get_redq_policy
from reinforce_policy import get_reinforce_policy
from a2c_policy import get_a2c_policy
from npg_policy import get_npg_policy
from trpo_policy import get_trpo_policy
from ppo_policy import get_ppo_policy

def load_policy(policy_type, env, policy_path, hidden_sizes):
    """
    根据策略类型加载策略模型，并从指定路径加载模型参数。
    
    参数：
        policy_type (str): 策略类型
        env (gym.Env): 强化学习环境
        policy_path (str): 模型参数路径
        hidden_sizes (list): 隐藏层大小
    
    返回：
        policy: 策略模型
        args: 策略模型的参数
    """
    if policy_type == "ddpg":
        policy, args = get_ddpg_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "td3":
        policy, args = get_td3_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "sac":
        policy, args = get_sac_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "redq":
        policy, args = get_redq_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    elif policy_type == "reinforce":
        policy, args = get_reinforce_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "a2c":
        policy, args = get_a2c_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "npg":
        policy, args = get_npg_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "trpo":
        policy, args = get_trpo_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    elif policy_type == "ppo":
        policy, args = get_ppo_policy(env, hidden_sizes=hidden_sizes)
        policy.load_state_dict(torch.load(policy_path, map_location=args.device)["model"])
    else:
        raise Exception("Unknown policy.")

    return policy, args

def simulate(task, policy_type, policy_path, hidden_sizes):
    """
    在指定的任务环境中运行指定的策略模型，并记录模拟过程的视频。
    
    参数：
        task (str): 任务环境名称
        policy_type (str): 策略类型
        policy_path (str): 策略模型参数路径
        hidden_sizes (list): 隐藏层大小
    """
    # 视频文件的名称前缀和存储文件夹
    video_name_prefix = f"{policy_type.upper()}_{task}"
    video_folder = os.path.join("videos", task, policy_type)

    # 创建环境并包装为录制视频的环境
    env = RecordVideo(
        env=gym.make(task, render_mode="rgb_array"),
        video_folder=video_folder,
        name_prefix=video_name_prefix,
        video_length=20000
    )
    observation, info = env.reset()

    # 加载策略模型
    policy, args = load_policy(policy_type=policy_type, env=env, policy_path=policy_path, hidden_sizes=hidden_sizes)
    print(f"从 {policy_path} 加载的代理")

    for step_index in range(2000):
        batch = Batch(obs=[observation], info=info)  # 构建Batch
        action = policy.forward(batch=batch, state=observation).act[0].detach().numpy()  # 获取动作

        observation, reward, terminated, truncated, info = env.step(action)  # 执行动作

        if terminated or truncated:  # 如果环境终止或截断，重置环境
            observation, info = env.reset()

    env.close()  # 关闭环境

if __name__ == '__main__':
    # 定义支持的策略类型及其隐藏层大小
    policies = {
        "ddpg": [256, 256],
        "td3": [256, 256],
        "sac": [256, 256],
        "redq": [256, 256],
        "reinforce": [64, 64],
        "a2c": [64, 64],
        "npg": [64, 64],
        "trpo": [64, 64],
        "ppo": [64, 64]
    }

    # 定义任务列表
    tasks = [
        "Ant-v4",
        "HalfCheetah-v4",
        "Hopper-v4",
        "Humanoid-v4",
        "HumanoidStandup-v4",
        "InvertedDoublePendulum-v4",
        "InvertedPendulum-v4",
        "Pusher-v4",
        "Reacher-v4",
        "Swimmer-v4",
        "Walker2d-v4"
    ]

    # 用户指定的策略类型和任务
    selected_policy = "sac"  # 示例：可以根据需要更改
    selected_task = "Humanoid-v4"  # 示例：可以根据需要更改
    selected_policy_path = "path/to/ppo_policy.pth"  # 示例：需要用户指定实际路径

    # 执行模拟
    simulate(
        task=selected_task,
        policy_type=selected_policy,
        policy_path=selected_policy_path,
        hidden_sizes=policies[selected_policy]
    )
