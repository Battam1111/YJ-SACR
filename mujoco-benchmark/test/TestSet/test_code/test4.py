import torch
import torch.nn as nn
import torch.optim as optim
from tianshou.policy import SACPolicy
# 定义简单的actor和critic网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x), torch.ones_like(self.fc3(x))  # 均值和标准差

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 创建环境和策略
state_dim = 3
action_dim = 2

actor = Actor(state_dim, action_dim)
critic1 = Critic(state_dim, action_dim)
critic2 = Critic(state_dim, action_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=0.001)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=0.001)

policy = SACPolicy(
    actor=actor,
    actor_optim=actor_optimizer,
    critic1=critic1,
    critic1_optim=critic1_optimizer,
    critic2=critic2,
    critic2_optim=critic2_optimizer,
    alpha=(0.2, torch.tensor([0.2], requires_grad=True), optim.Adam([torch.tensor([0.2], requires_grad=True)], lr=0.001))
)

# 测试前向传播
batch = Batch(obs=torch.randn(10, state_dim))
output = policy.forward(batch)
print(output)

# 测试学习
buffer = ReplayBuffer(size=100)
for _ in range(10):
    buffer.add(
        obs=torch.randn(state_dim),
        act=torch.randn(action_dim),
        rew=np.random.randn(),
        obs_next=torch.randn(state_dim),
        done=False,
    )

batch, _ = buffer.sample(10)
result = policy.learn(batch)
print(result)
