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


class SchedPolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(SchedPolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, mask=None):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        # ------ 4. mask 已排课程 ------
        logits = self.fc3(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs

class SchedValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(SchedValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)   # [B, 1]

class Scheduler:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, lmbda, eps, gamma, device, weights_dir):
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device
        self.weights_dir = weights_dir

        # 单独一个 actor 和一个 critic
        self.actor = SchedPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = SchedValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

    def save_model(self, path=None):
        if path is None:
            path = self.weights_dir
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor_scheduler.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic_scheduler.pth"))

    def load_model(self, path=None):
        if path is None:
            path = self.weights_dir
        actor_path = os.path.join(path, "actor_scheduler.pth")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
        critic_path = os.path.join(path, "critic_scheduler.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))

    def take_action(self, states, masks=None):
        """
        state: 1D np.array, scheduler 的全局状态
        mask:  1D np.array, 已排 / 不可选的课程位置为 0，可选为 1
        返回: action_idx, action_prob_vec（或你只要 log_prob 也行）
        """
        actions = []
        action_probs = []

        for i, state in enumerate(states):
            s = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
            m = torch.tensor(np.array([masks[i]]), dtype=torch.float32, device=self.device) if masks is not None else None

            probs = self.actor(s, m)   # [1, action_dim]
            probs = torch.clamp(probs, 1e-8, 1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()     # [1]
            
            actions.append(action.item())
            action_probs.append(probs.detach().cpu().numpy()[0])
        return actions, action_probs

    def update(self, buffer):
        """
        buffer: dict，结构和 MAPPO 每个智能体的 transition_dict 对齐：
          buffer['states']       : [T, state_dim]
          buffer['actions']      : [T] (int)
          buffer['action_probs'] : [T, action_dim] (旧策略下的概率分布)
          buffer['rewards']      : [T]
          buffer['dones']        : [T] (0/1)
          buffer['next_states']  : [T, state_dim]
        """
        if len(buffer['states']) == 0:
            # 防止空 episode
            return 0.0, 0.0, 0.0

        # ------- 1. 转成 tensor -------
        states = torch.tensor(
            np.array(buffer['states']),
            dtype=torch.float32,
            device=self.device
        )                                # [T, state_dim]

        next_states = torch.tensor(
            np.array(buffer['next_states']),
            dtype=torch.float32,
            device=self.device
        )                                # [T, state_dim]

        actions = torch.tensor(
            np.array(buffer['actions']),
            dtype=torch.long,
            device=self.device
        ).view(-1, 1)                    # [T, 1]

        old_probs = torch.tensor(
            np.array(buffer['action_probs']),
            dtype=torch.float32,
            device=self.device
        )                                # [T, action_dim]

        rewards = torch.tensor(
            np.array(buffer['rewards']),
            dtype=torch.float32,
            device=self.device
        ).view(-1, 1)                    # [T, 1]

        dones = torch.tensor(
            np.array(buffer['dones']),
            dtype=torch.float32,
            device=self.device
        ).view(-1, 1)                    # [T, 1]

        # ------- 2. Critic：算 value、TD target、advantage -------
        values = self.critic(states)              # [T, 1]
        with torch.no_grad():
            next_values = self.critic(next_states)       # [T, 1]
            td_target = rewards + self.gamma * next_values * (1 - dones)  # [T, 1]
            td_delta = td_target - values                                  # [T, 1]

            advantages = compute_advantage(
                self.gamma, self.lmbda, td_delta.squeeze(-1)
            )   # [T] on CPU
            # 标准化一下优势，防止尺度太大
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.to(self.device).view(-1, 1)    # [T, 1]

        # Critic loss = MSE(V, target)
        critic_loss = F.mse_loss(values, td_target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ------- 3. Actor：PPO clipped objective -------
        current_probs = self.actor(states)          # [T, action_dim]
        # 取当前策略对已执行动作的概率
        action_probs = current_probs.gather(1, actions)              # [T, 1]
        old_action_probs = old_probs.gather(1, actions)              # [T, 1]

        log_probs = torch.log(action_probs + 1e-8)                   # [T, 1]
        old_log_probs = torch.log(old_action_probs + 1e-8).detach()  # [T, 1]

        ratio = torch.exp(log_probs - old_log_probs)                 # [T, 1]

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

        actor_loss = -torch.mean(torch.min(surr1, surr2))
        entropy_val = compute_entropy(current_probs)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy_val

#  策略网络(Actor)
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, mask=None):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        # ------ 4. mask 已排课程 ------
        logits = self.fc3(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs

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

    def take_action(self, state_per_agent, state_mask=None):
        actions = []
        action_probs = []
        for i, actor in enumerate(self.actors):
            s = torch.tensor(np.array([state_per_agent[i]]), dtype=torch.float).to(self.device)
            m = torch.tensor(np.array([state_mask[i]]), dtype=torch.float).to(self.device) if state_mask is not None else None
            probs = actor(s, m)
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
