import torch
import math

# 定义对数概率密度函数及其一阶和二阶导数
def log_prob(y, loc, scale):
    term1 = (0.5 * torch.log((1 + y) / (1 - y)) - loc) ** 2 / (-2 * (scale ** 2))
    term2 = -torch.log(1 - y ** 2)
    term3 = -0.5 * torch.log(2 * torch.tensor(math.pi) * scale ** 2)
    return term1 + term2 + term3

def log_prob_prime(y, loc, scale):
    term1 = -((1 + y) / (1 - y)).reciprocal() - ((1 - y) / (1 + y)).reciprocal()
    term2 = 2 * (0.5 * torch.log((1 + y) / (1 - y)) - loc) / ((scale ** 2) * (1 - y ** 2))
    term3 = -2 * y / (1 - y ** 2)
    return term1 * term2 + term3

def log_prob_prime2(y, loc, scale):
    term1 = (1 - y ** 2).reciprocal()
    term2 = -2 * (0.5 * torch.log((1 + y) / (1 - y)) - loc) * (2 * y / (scale ** 2) * (1 - y ** 2) ** 2)
    term3 = -2 / (1 - y ** 2)
    return term1 * term2 + term3

# 初始猜测值
y = torch.tensor(0.0, requires_grad=True)
loc = torch.tensor(0.0, requires_grad=True)
scale = torch.tensor(1.0, requires_grad=True)

# 牛顿-拉夫森迭代
for _ in range(10):  # 迭代次数可以调整
    with torch.no_grad():
        f_prime = log_prob_prime(y, loc, scale)
        f_prime2 = log_prob_prime2(y, loc, scale)
        y -= f_prime / f_prime2

y_star = y.item()
print(f"The mode of the distribution is: {y_star}")

# 计算对应的概率密度值
log_prob_value = log_prob(torch.tensor(y_star), loc, scale).item()
pdf_value = math.exp(log_prob_value)
print(f"The probability density at the mode is: {pdf_value}")
