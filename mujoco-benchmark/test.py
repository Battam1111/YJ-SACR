# import torch
# import numpy as np
# import csv

# mean = torch.tensor(
#     [[0],
# ]
# )

# std = torch.tensor(
#     [[1],
# ]
# )

# eps = np.finfo(np.float32).eps.item()

# # 生成均匀分布的采样点
# y0 = 0.9999
# sample_range = torch.linspace(-y0, y0, 2000).to(mean.device)

# # 扩展采样点的维度，以匹配批量大小和动作维度
# sample_range = sample_range.unsqueeze(0).unsqueeze(0).expand(std.shape[0], std.shape[1], 2000)

# # 将 sample_range 保存到 CSV 文件
# with open('sample_range.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["sample_range"])
#     for row in sample_range:
#         writer.writerow(row.tolist())

# # 计算每个样本的概率密度（原版）
# log_term = (0.5 * torch.log((1 + sample_range) / (1 - sample_range) + eps) - mean.unsqueeze(2)) ** 2 / (-2 * (std.unsqueeze(2) ** 2))
# log_probs = (1 / (1 - sample_range ** 2)) * (1 / (((2 * torch.tensor(torch.pi) * std.unsqueeze(2) ** 2)) ** 0.5)) * torch.exp(log_term)



# # 计算每个样本的概率密度(与公式不同版)
# # log_probs = (-0.5 * ((torch.log((1 + sample_range) / (1 - sample_range) + eps) - mean.unsqueeze(2)) / std.unsqueeze(2)) ** 2 - 
# #             torch.log(std.unsqueeze(2) * torch.sqrt(torch.tensor(2 * np.pi)).to(mean.device)) -
# #             torch.log(1 - sample_range ** 2 + eps))

# # 将 log_probs 保存到 CSV 文件
# with open('log_probs.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["log_probs"])
#     for batch in log_probs:
#         for row in batch:
#             writer.writerow(row.tolist())

# # 选择最优的采样点索引
# best_sample_idx = torch.argmax(log_probs, dim=2)

# # 将 best_sample_idx 保存到 CSV 文件
# with open('best_sample_idx.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["best_sample_idx"])
#     for row in best_sample_idx:
#         writer.writerow(row.tolist())

# # 选择最优的采样点
# best_sample = torch.gather(sample_range, 2, best_sample_idx.unsqueeze(2)).squeeze(2)

# # 将 best_sample 保存到 CSV 文件
# with open('best_sample.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["best_sample"])
#     for row in best_sample:
#         writer.writerow(row.tolist())


import mujoco
import gymnasium as gym
import logging

logging.basicConfig(level=logging.DEBUG)

def test_mujoco_env():
    try:
        env = gym.make('HalfCheetah-v4', render_mode='human')
        env.reset()
        env.render()
        print("Mujoco and Gymnasium are installed correctly.")
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Detailed error information")

if __name__ == "__main__":
    test_mujoco_env()

