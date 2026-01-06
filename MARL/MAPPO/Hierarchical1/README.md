# 分层MARL（Hierarchical MARL）

- 上层：调度 agent（Scheduler）——专门学“下一门排谁”（课程顺序）。

- 下层：课程 agent（Course agents）——给定当前被选中的课程，选它的 (教室, 时间)。

- 下层用 MAPPO（参数共享 Actor + 中心化 Critic）+ 动作 mask 来处理每门课不同的候选集合。

## 层次化 Multi-Agent（Scheduler + Course Agents）

### 1 上层：课程顺序调度 agent（Scheduler）

状态：全局课表状态 + 哪些课还没排 + 这些课的特征（学生人数、优先级、难排程度估计等）。

动作：从“未排课程集合”里选出“下一门要排的课 course_i”。

算法：普通单智能体 RL 就行：

DQN（离散动作）、PPO、A2C 都可以

一轮 episode 结束后，根据整张课表的 penalty 得到 R，用回传更新 scheduler。

这个 agent 负责学“先排哪些课比较好”（例如先排大课、冲突多的课等）。

### 2 下层：课程 agents（Course-level MAPPO）

有很多门课程，每门课程是一个 agent，但共享一套 Actor 参数（parameter sharing），Critic 是中心化的（CTDE 思想）。

Actor（对课程 i）

输入：

当前被 scheduler 选中的课程 i 的特征（需求教室类型、学生数、老师可用时间等）

当前已排课表的局部/全局摘要（可以用 attention / GNN 编）。

输出：

为课程 i 的所有候选 (room, slot) 组合打分。

用 mask 把非法或超出候选集合的动作位设置为 -inf。

Critic

输入整个课表状态（所有已排 + 未排课程信息）。

输出一个全局 value，用于 advantage 估计。
→ 典型 MAPPO 结构，只不过你是 异构动作空间 + mask。

训练流程（简化版）

scheduler 选一门课 i。

激活课程 agent i 的 Actor，选择该课的 (room, slot)。

环境更新课表、检查约束。

重复直到所有课排完，得到总 penalty → 定义全局回报 R。

用 R 同时更新 scheduler 的 policy 和所有课程 Actor + 中心 Critic。

