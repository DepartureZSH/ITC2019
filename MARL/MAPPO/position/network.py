import os
import torch
import torch.nn.functional as F
import numpy as np

def compute_entropy(probs):
    dist = torch.distributions.Categorical(probs)
    return dist.entropy().mean().item()

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

#  策略网络(Actor)
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)

# 全局价值网络(CentralValueNet)
# 输入: 所有智能体的状态拼接 (team_size * state_dim)
# 输出: 对每个智能体的价值估计 (team_size维向量)
class CentralValueNet(torch.nn.Module):
    def __init__(self, total_state_dim, hidden_dim, team_size):
        super(CentralValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, team_size)  # 输出为每个智能体一个价值

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)  # [batch, team_size]


class MAPPO:
    def __init__(self, team_size, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, lmbda, eps, gamma, device, weights_dir):
        self.team_size = team_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device
        self.weights_dir = weights_dir

        # 为每个智能体一个独立的actor
        self.actors = [PolicyNet(state_dim, hidden_dim, action_dim).to(device)
                       for _ in range(team_size)]

        # 一个全局critic，输入为所有智能体状态拼接
        self.critic = CentralValueNet(team_size * state_dim, hidden_dim, team_size).to(device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), actor_lr) 
                                 for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

    def save_model(self, path=None):
        if path is None:
            path = self.weights_dir
        if not os.path.exists(path):
            os.makedirs(path)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(path, f"actor_{i}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def load_model(self, path=None):
        if path is None:
            path = self.weights_dir
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"actor_{i}.pth")
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path))
        critic_path = os.path.join(path, "critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))

    def take_action(self, state_per_agent):
        actions = []
        action_probs = []
        for i, actor in enumerate(self.actors):
            s = torch.tensor(np.array([state_per_agent[i]]), dtype=torch.float).to(self.device)
            probs = actor(s)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            actions.append(action.item())
            action_probs.append(probs.detach().cpu().numpy()[0])
        return actions, action_probs

    def update(self, transition_dicts, state_dim):
        # 拼接所有智能体的数据，用于全局critic
        # 首先统一长度T，假设所有智能体长度相同（因为同步环境步）
        T = len(transition_dicts[0]['states'])
        # 将所有智能体在同一时间步的state拼接起来，得到 [T, team_size*state_dim]
        states_all = []
        next_states_all = []
        for t in range(T):
            concat_state = []
            concat_next_state = []
            for i in range(self.team_size):
                concat_state.append(transition_dicts[i]['states'][t])
                concat_next_state.append(transition_dicts[i]['next_states'][t])
            states_all.append(np.concatenate(concat_state))
            next_states_all.append(np.concatenate(concat_next_state))
            
        states_all = np.array(states_all)
        next_states_all = np.array(next_states_all)
        states_all = torch.tensor(states_all, dtype=torch.float).to(self.device)  # [T, team_size*state_dim]
        next_states_all = torch.tensor(next_states_all, dtype=torch.float).to(self.device) # [T, team_size*state_dim]

        rewards_all = torch.tensor([[transition_dicts[i]['rewards'][t] for i in range(self.team_size)] 
                                     for t in range(T)], dtype=torch.float).to(self.device) # [T, team_size]
        dones_all = torch.tensor([[transition_dicts[i]['dones'][t] for i in range(self.team_size)] 
                                   for t in range(T)], dtype=torch.float).to(self.device) # [T, team_size]

        # 从critic计算价值和TD-target
        values = self.critic(states_all) # [T, team_size]    
        next_values = self.critic(next_states_all) # [T, team_size]
        td_target = rewards_all + self.gamma * next_values * (1 - dones_all) # [T, team_size]
        td_delta = td_target - values # [T, team_size]

        # 为每个智能体计算其优势
        advantages = []
        for i in range(self.team_size):
            adv_i = compute_advantage(self.gamma, self.lmbda, td_delta[:, i])
            advantages.append(adv_i.to(self.device))  # [T]

        # 更新critic
        # critic的loss是所有智能体的均方误差平均
        critic_loss = F.mse_loss(values, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新每个智能体的actor
        action_losses = []
        entropies = []

        for i in range(self.team_size):
            states = torch.tensor(np.array(transition_dicts[i]['states']), dtype=torch.float).to(self.device)
            actions = torch.tensor(np.array(transition_dicts[i]['actions'])).view(-1, 1).to(self.device)
            old_probs = torch.tensor(np.array(transition_dicts[i]['action_probs']), dtype=torch.float).to(self.device)

            current_probs = self.actors[i](states) # [T, action_dim]
            log_probs = torch.log(current_probs.gather(1, actions))
            old_log_probs = torch.log(old_probs.gather(1, actions)).detach()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages[i].unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages[i].unsqueeze(-1)

            action_loss = torch.mean(-torch.min(surr1, surr2))
            entropy_val = compute_entropy(current_probs)

            self.actor_optimizers[i].zero_grad()
            action_loss.backward()
            self.actor_optimizers[i].step()

            action_losses.append(action_loss.item())
            entropies.append(entropy_val)

        return np.mean(action_losses), critic_loss.item(), np.mean(entropies)
