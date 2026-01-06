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

class SchedulerPolicy(torch.nn.Module):
    def __init__(self, sched_state_dim, hidden_dim, max_courses, alpha=2.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(sched_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, max_courses)

        self.num_courses = max_courses
        self.alpha = alpha  # 先验权重，越大越“听 agent.value 的话”

        # 可选：把网络初始化得很小，这样一开始几乎完全等于先验
        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x, mask=None):
        """
        x: [B, sched_state_dim]
        假设 x 前 num_courses 维是各个课程的 value。
        """

        B = x.size(0)

        # ------ 1. 网络本身的 logits ------
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits_net = self.fc3(h)  # [B, num_courses]

        # ------ 2. 从 state 中取出 value，构造先验 logits ------
        # values: [B, num_courses]
        values = x[:, :self.num_courses]

        # 标准化一下，避免尺度影响太大
        v_mean = values.mean(dim=1, keepdim=True)
        v_std = values.std(dim=1, keepdim=True)
        v_norm = (values - v_mean) / (v_std + 1e-6)   # 有可能某一刻所有 value 一样，加 eps 防止除 0

        # value 越小，logit 越大 → softmax 更偏向 value 小的课程
        prior_logits = - self.alpha * v_norm          # [B, num_courses]

        # ------ 3. 合并：网络学习的修正 + 先验 ------
        logits = logits_net + prior_logits

        # ------ 4. mask 已排课程 ------
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs

class SchedulerValue(torch.nn.Module):
    def __init__(self, sched_state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(sched_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)   # [B,1]

class Scheduler:
    def __init__(self, sched_state_dim, hidden_dim, num_courses,
                 actor_lr, critic_lr, gamma, lmbda, eps, device, weights_dir):
        self.actor = SchedulerPolicy(sched_state_dim, hidden_dim, num_courses).to(device)
        self.critic = SchedulerValue(sched_state_dim, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device
        self.weights_dir = weights_dir
    
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

    def take_action(self, sched_state, mask):
        # sched_state: numpy 1D or 2D
        s = torch.tensor(sched_state, dtype=torch.float, device=self.actor.fc1.weight.device).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.float, device=self.actor.fc1.weight.device).unsqueeze(0)
        probs = self.actor(s, m)      # [1, max_courses]
        probs = torch.clamp(probs, 1e-8, 1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()         # 一个 course 索引

        probs_np = probs.detach().cpu().numpy()[0]
        log_prob = np.log(probs_np[action.item()] + 1e-8)

        return action.item(), log_prob
    
    def update(self, buffer):
        # buffer : { 'states':[], 'actions':[], 'action_probs':[], 'rewards':[], 'dones':[], 'next_states':[] }
        # 和你 MAPPO.update 里面一样，用 advantage + clip PPO 写就行
        # 将所有智能体在同一时间步的state拼接起来，得到 [T, team_size*state_dim]
        if len(buffer['states']) == 0:
            # 防止空 episode（极端情况），直接不更新
            return 0.0, 0.0, 0.0

        # ------- 1. 转成 tensor -------
        states = torch.tensor(
            np.array(buffer['states']),
            dtype=torch.float32,
            device=self.device
        )                        # [T, sched_state_dim]

        next_states = torch.tensor(
            np.array(buffer['next_states']),
            dtype=torch.float32,
            device=self.device
        )                        # [T, sched_state_dim]

        actions = torch.tensor(
            np.array(buffer['actions']),
            dtype=torch.long,
            device=self.device
        ).view(-1, 1)            # [T, 1]

        old_action_probs = torch.tensor(
            np.array(buffer['action_probs']),
            dtype=torch.float32,
            device=self.device
        ).view(-1, 1)            # [T, 1]

        rewards = torch.tensor(
            np.array(buffer['rewards']),
            dtype=torch.float32,
            device=self.device
        ).view(-1, 1)            # [T, 1]

        dones = torch.tensor(
            np.array(buffer['dones']),
            dtype=torch.float32,
            device=self.device
        ).view(-1, 1)            # [T, 1]

        # ------- 2. Critic：算 value、TD target、advantage -------
        values = self.critic(states)          # [T, 1]
        with torch.no_grad():
            next_values = self.critic(next_states)   # [T, 1]
            td_target = rewards + self.gamma * next_values * (1 - dones)  # [T, 1]
            td_delta = td_target - values                                  # [T, 1]

            # compute_advantage 期望输入 1D tensor
            advantages = compute_advantage(
                self.gamma, self.lmbda, td_delta.squeeze(-1)
            )  # [T] on CPU

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.to(self.device).view(-1, 1)        # [T]

        # Critic loss = MSE(V, target)
        critic_loss = F.mse_loss(values, td_target.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # ------- 3. Actor：PPO clipped objective -------
        # 当前策略下的概率
        current_probs = self.actor(states)          # [T, num_courses]
        action_probs = current_probs.gather(1, actions) # [T, 1]
        log_probs = torch.log(action_probs + 1e-8)

        ratio = torch.exp(log_probs - old_action_probs)  # [T, 1]

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

        actor_loss = -torch.mean(torch.min(surr1, surr2))
        entropy_val = compute_entropy(current_probs)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # 返回三个标量，和你 MAPPO.update 的风格一致
        return actor_loss.item(), critic_loss.item(), entropy_val

        # return np.mean(action_losses), critic_loss.item(), np.mean(entropies)


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

    def take_action(self, cid_action, state_agent):
        actions = []
        action_probs = []
        # for i, actor in enumerate(self.actors):
        actor = self.actors[cid_action]
        s = torch.tensor(np.array(state_agent), dtype=torch.float).to(self.device)
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
