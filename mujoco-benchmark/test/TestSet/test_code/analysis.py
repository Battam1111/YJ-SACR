import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def save_plot(fig, filename):
    """保存图像到指定路径"""
    # 获取目录路径
    directory = os.path.dirname(filename)
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 保存图像
    fig.savefig(filename)

# 读取CSV文件
file_path = 'mujoco-benchmark/test/TestSet/test_4/Ant-af0.3/mean.csv'  # 替换为你的CSV文件路径
save_path = 'mujoco-benchmark/test/TestSet/Plots/Ant-af0.3-mean'  # 替换为你希望保存图像的路径
data = pd.read_csv(file_path, header=None)

# 只处理后50%的数据
half_index = len(data) // 2
data = data.iloc[half_index:]

# 查看数据基本信息
print(data.head())
print(data.describe())

# 绘制时间序列图
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(data.shape[1]):
    ax.plot(data.index, data[i], label=f'Action {i+1}')

ax.set_xlabel('Sample Index')
ax.set_ylabel('Action Value')
ax.set_title('Time Series of Actions')
ax.legend()
save_plot(fig, os.path.join(save_path, 'time_series.png'))
plt.close(fig)  # 关闭图像以释放内存

# 绘制直方图
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(data.shape[1]):
    ax.hist(data[i], bins=30, alpha=0.5, label=f'Action {i+1}')

ax.set_xlabel('Action Value')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Actions')
ax.legend()
save_plot(fig, os.path.join(save_path, 'histogram.png'))
plt.close(fig)  # 关闭图像以释放内存

# 绘制散点图（如果数据维度较高，可以使用PCA进行降维）
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA of Actions')
save_plot(fig, os.path.join(save_path, 'pca_scatter.png'))
plt.close(fig)  # 关闭图像以释放内存

# 计算每个动作（列）的平均值
means = data.mean()

# 绘制平均值条形图
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(data.shape[1]), means, tick_label=[f'Action {i+1}' for i in range(data.shape[1])])

ax.set_xlabel('Action')
ax.set_ylabel('Average Value')
ax.set_title('Average Value of Each Action')
save_plot(fig, os.path.join(save_path, 'average_values.png'))
plt.close(fig)  # 关闭图像以释放内存