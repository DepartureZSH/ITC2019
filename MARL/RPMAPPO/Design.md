## 1. 上层：Scheduler（Q-Learning）

这是“排课顺序控制器”，只管**顺序**，不管具体选哪个 room–time。

### Observation（调度层观测）

* 对 Scheduler 来说，一个 episode 里它要多次调整课程顺序。

* 在每一轮探索中，调度层看到的是：

  > 课程在当前排课顺序中的 **位置索引**。

* 实现上可以理解为：对第 (i) 门课（或第 (i) 个位置），其 observation 就是当前的排序位置 `pos(i)`，Q-table 以这个索引作为行。

### Mask（调度层 mask）

* `scheduler_mask`：长度 = 课程数的 0/1 向量。

* 语义：

  > `mask[c] = 1` 表示课程 (c) 在本轮探索中可以被调整顺序；
  > `mask[c] = 0` 表示课程 (c) 在本轮不参与调整（已经锁定或不应移动）。

* Scheduler 只对 `mask == 1` 的课程输出动作。

### Action（调度层动作）

你现在的设计是：

> **Action：本轮每个课程在排课顺序中的位置。**

也就是：

* 对于每个允许调整的课程 (c)，Scheduler 输出一个目标位置（或新的 index），用来更新本轮的排课顺序；
* 可以理解成：**“在当前排序里，把课程 (c) 放到第 (a_c) 个位置上”**，或者实现上是与目标位置课程做 swap / 插入。

所以在论文里可以写成：

> 对每个可调课程，调度层动作为一个离散位置索引，表示该课程在本轮探索中应当被放置到的顺序位置。

### Reward（调度层奖励）

你给的是：

> Reward：每轮探索中的最佳 result 对应的 reward。

也就是：

* 在一轮调度探索中，Scheduler 会根据当前 Q-table 生成若干候选顺序（或多次微调同一顺序），每次都交给 Selector+环境完整排一次课；
* 这一轮中得到的所有 timetable 里，**挑出 penalty 最低 / 最可行的那一个**；
* 用这个“最佳 timetable”的指标（比如 (-)TotalCost 或你前面设计的组合）作为本轮 Scheduler 的 reward，去更新 Q-table。

论文表达可以写成：

> After each scheduling exploration round, we evaluate all candidate timetables produced by the current scheduler policy and define the scheduler reward as the quality (negative weighted cost) of the best timetable found in that round. This scalar reward is then used to update the Q-table entries associated with the scheduling decisions of that round.

---

## 2. 下层：Selector（MAPPO）

这是“选课时具体 room–time 动作的策略层”，你叫 Selector，用的是 MAPPO。

### Observation（选择层观测）

你现在的定义是：

> Observation：课程在当前排课顺序的位置后面的硬约束情况（hard constraints 和 room constraints）。

也就是说，对当前要排的课程 (\ell)，Selector 看到的不是完整 timetable，而是：

* 基于 **当前顺序和已经排过的课程**，环境分析出：

  * 后续时间段上该课程可能遇到的 teacher 冲突、room 冲突、不可用时段等；
  * 对不同候选动作（room–time 组合）产生的硬约束相关特征。
* 然后把这些特征编码成一个向量，作为 Selector 的 observation。

论文里可以抽象为：

> For each course being scheduled at a given position, the Selector observes a feature vector summarizing the hard feasibility of its remaining room–time options, including potential teacher conflicts, room availability constraints, and other hard constraints induced by already scheduled courses and room calendars.

（具体特征细节你可以在附录里展开，这里保持抽象就行。）

### Mask（选择层 mask）

你保留了和前几版一样的 mask 逻辑：

> Mask：标记动作合法区域。

* `agent.masked_actions` 是长度 = 候选动作数的 0/1 向量；
* `1` 表示该 room–time 组合在当前部分课表下 **满足所有 hard constraints**；
* `0` 表示该动作会违反硬约束，必须被 policy 屏蔽。

### Action（选择层动作）

你现在的采样逻辑是：

```python
probs = (agent.probs + 1e-6) * agent.masked_actions
probs = np.array(probs, dtype=np.float64)
probs = probs / np.sum(probs)
action_ind = self.agents[i]._action_space.sample(probability=probs)
action = self.agents[i].action_space[action_ind]
```

也就是：

1. Selector 的 actor 网络输出 `agent.probs`：对每个候选动作的原始概率；
2. 乘上 `masked_actions` 把非法动作清零，再加一个很小的数防止全零；
3. 归一化得到真正的分布；
4. 采样动作索引 `action_ind`，映射到具体的 `(room, time)` 组合。

论文可以写：

> The Selector’s policy network outputs a probability distribution over all candidate room–time assignments for the current course. This distribution is then element-wise multiplied by the hard-constraint mask, renormalized, and a single action (concrete room–time option) is sampled from the resulting masked distribution.

### Reward（选择层奖励）

你给的是：

```python
r1 = len(self._assignment)
for agent in self.agents:
    if agent.id in self._assignment:
        r2 = - agent.max_penalty
    else:
        r2 = np.max(agent.action_penalty) - r1 * Total_cost / len(self.agents)
    reward = r2
    rewards.append(reward)
```

可以这样理解：

* `self._assignment`：当前 **无可行动作 / 被判定为无法分配** 的课程集合；

  * 所以 `r1 = len(self._assignment)` 是“本轮失败课程数”。

* 对每一个课时 agent：

  * 如果它在 `_assignment` 里（即 **最终没排上**）：

    * `r2 = - agent.max_penalty`：
      → 给一个较大的负奖励（罚得比较重），鼓励策略避免让该课走向“无可用动作”的局面；
  * 否则（成功分配）：

    * `r2 = np.max(agent.action_penalty) - r1 * Total_cost / len(self.agents)`：
      → 第一项 `np.max(agent.action_penalty)` 是该课动作空间的一个“尺度”；
      → 第二项 `r1 * Total_cost / len(self.agents)` 把全局成本和未分配数压缩成对每个已分配课程共享的惩罚；
      → 实际上是一个“反向的 cost 信号”：总成本越高、未分配越多，这一项越负。

* 最终每个课时的 reward = `r2`。

你可以在论文中抽象成：

> At the Selector level, we use a per-course reward that combines a strong penalty for unscheduled courses with a global cost-based signal. Let (|\mathcal{U}|) denote the number of unscheduled courses in the current timetable and (\text{TotalCost}) the weighted sum of time, room, and distribution penalties. For each course (\ell), we assign
> [
> r_\ell =
> \begin{cases}
>
> * P_\ell^{\max}, & \text{if } \ell \in \mathcal{U},[0.3em]
>   P_\ell^{\max} - \dfrac{|\mathcal{U}|}{|\mathcal{L}|} \cdot \text{TotalCost}, & \text{otherwise},
>   \end{cases}
>   ]
>   where (P_\ell^{\max}) is a scale factor derived from the penalty range of (\ell)’s action space and (|\mathcal{L}|) is the total number of courses. Thus, unscheduled courses receive a large negative reward, while scheduled courses share a global penalty term that grows with both the total cost and the number of unscheduled courses.

