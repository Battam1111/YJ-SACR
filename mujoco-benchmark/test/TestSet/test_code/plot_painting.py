import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
    """从CSV文件读取数据"""
    data = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(item) for item in row])
    return np.array(data)

# 读取CSV文件
file_path_uniform_original = '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/uniform_actions.csv'
file_path_uniform_modified = '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/uniform_actions_f.csv'

data_uniform_original = read_csv(file_path_uniform_original)
data_uniform_modified = read_csv(file_path_uniform_modified)

# 选择一种动作（例如选择第一个动作）
action_index = 0
action_uniform_original = data_uniform_original[:, action_index]
action_uniform_modified = data_uniform_modified[:, action_index]

# 绘制图表
plt.figure(figsize=(10, 6))

# 绘制原始数据
plt.plot(action_uniform_original, label='Uniform Actions (Original)', color='blue')

# 绘制修改后的数据
plt.plot(action_uniform_modified, label='Uniform Actions (Modified)', color='red', linestyle='dashed')

# 添加图表标题和标签
plt.title(f'Action {action_index} - Uniform Sampling Comparison')
plt.xlabel('Sample Index')
plt.ylabel('Action Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
