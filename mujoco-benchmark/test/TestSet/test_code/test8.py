import torch
import math
import numpy as np
from scipy.optimize import minimize

# 定义对数概率密度函数
def log_prob(y, loc, scale):
    log_term = (0.5 * np.log((1 + y) / (1 - y)) - loc) ** 2 / (-2 * (scale ** 2))
    log_probs = -np.log(1 - y ** 2) - 0.5 * np.log(2 * np.pi * scale ** 2) + log_term
    return log_probs

# 目标函数（需要最大化对数概率密度函数）
def objective(y, loc, scale):
    return -log_prob(y, loc, scale)  # 最小化负的对数概率密度

# 初始猜测值
y0 = 0.0
loc = 0.0
scale = 1.0

# 使用 SciPy 优化库找到使对数概率密度最大的点
result = minimize(objective, y0, args=(loc, scale), bounds=[(-1 + 1e-5, 1 - 1e-5)])
y_star = result.x[0]

print(f"The mode of the distribution is: {y_star}")

# 计算对应的概率密度值
log_prob_value = log_prob(y_star, loc, scale)
pdf_value = np.exp(log_prob_value)
print(f"The probability density at the mode is: {pdf_value}")
