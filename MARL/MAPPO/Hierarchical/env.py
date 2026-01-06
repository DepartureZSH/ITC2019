import pathlib
import yaml
import copy
import torch
import json
import numpy as np
from MARL.MAPPO.position.agents import agent_class
from MARL.MAPPO.position.constraints import HardConstraints, SoftConstraints
from math import inf
from tqdm import tqdm
from gymnasium import spaces

folder = pathlib.Path(__file__).parent.resolve()

dead_reward = -999

class CustomEnvironment:
    def __init__(self, reader, obs_shape=32, action_shape=32):
        self.reader = reader
        self.iter = 0
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        
        self.hard_constrains = self.reader.distributions['hard_constraints']
        self.soft_constrains = self.reader.distributions['soft_constraints']
        self.Hard_validator = HardConstraints()
        self.Soft_validator = SoftConstraints()

        self._agents = []
        self.agents = [] # classes
        self.cid2ind = {}
        i = 0
        for key, each in self.reader.classes.items():
            self.agents.append(agent_class(each, self.obs_shape))
            self.cid2ind[key] = i
            self._agents.append(key)
            i += 1
        self.travel = self.reader.travel
        self.Hard_validator.setTravel(self.travel)
        self.Hard_validator.sefnrDays(self.reader.nrDays)
        self.Hard_validator.sefnrWeeks(self.reader.nrWeeks)
        self.Hard_validator.setCid2ind(self.cid2ind)
        self.Soft_validator.setTravel(self.travel)
        self.Soft_validator.sefnrDays(self.reader.nrDays)
        self.Soft_validator.sefnrWeeks(self.reader.nrWeeks)
        self.Soft_validator.setCid2ind(self.cid2ind)
        self._assignment = []
        self._sched = []

        # self.timeTable_matrix = self.reader.timeTable_matrix
        self.rooms = copy.deepcopy(self.reader.rooms) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}

        # self._action_space = spaces.Box(low=0, high=len(self.agents), shape=(len(self.agents),), dtype=np.float64)  # Placeholder
        # self.observation_spaces = spaces.Box(low=-1e9, high=1e9, shape=(len(self.agents),), dtype=np.float64)

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self._assignment = []
        self._sched = []
        self.rooms = copy.deepcopy(self.reader.rooms)
        observations = {}
        for agent in self.agents:
            agent.candidate = None
            agent.action = None
            agent.penalty = inf
            agent.choice = 0
            agent.masked_actions = [1 for _ in range(len(agent.action_space))]
            observations[agent.id] = self.agent_observe(agent.id)
            agent.action_space = agent._actions()
            agent.value = len(agent.action_space)
            agent.observe_space = np.array([len(agent.action_space)/1 for _ in range(self.obs_shape)], dtype=np.float64)
        not_assignment = [agent.id for agent in self.agents]
        return observations, not_assignment
    
    def get_unassigned_courses(self):
        # 未排的课程列表
        return [cid for cid in self._agents if cid in self._assignment]
    
    def get_unsched_courses(self):
        # 未排的课程列表
        return [cid for cid in self._agents if cid not in self._sched]

    def get_scheduler_obs(self):
        """
        构造给 Scheduler 的全局状态向量
        简单版：拼一些统计量就行，比如：
         - 每门课一个 scalar：agent.value 或者 penalty 估计
         - 每门课一个已/未排 flag
        """
        values = []
        assign = []
        sched = []
        for cid in self._agents:
            i = self.cid2ind[cid]
            if cid not in self._sched:
                values.append(self.agents[i].value)                 # e.g. len(action_space) or dynamic score
            else:
                values.append(0.0)

        # 简单拼接
        state = np.concatenate([np.array(values, dtype=np.float64)])
        return state

    def get_scheduler_mask(self):
        """
        返回长度 = num_courses 的 mask:
        - 只要还有未排的 value=1 课程，就只允许选这些课程；
        - 当所有 value=1 课程都排完后，才允许选其他课程。
        """
        mask = []
        # 先找出“未排且 value=1”的课程
        urgent_courses = []
        for cid in self._agents:
            if cid in self._sched:
                continue
            i = self.cid2ind[cid]
            if self.agents[i].value == 1:
                urgent_courses.append(cid)

        # 是否存在“必须优先排”的课程
        has_urgent = len(urgent_courses) > 0

        for cid in self._agents:
            # 已经排过的一律不能选
            if cid in self._sched:
                mask.append(0.0)
                continue

            i = self.cid2ind[cid]
            v = self.agents[i].value

            if has_urgent:
                # 还有 value=1 的课没排 → 只允许选这些
                if v == 1:
                    mask.append(1.0)
                else:
                    mask.append(0.0)
            else:
                # 没有 urgent 课了 → 所有未排的都可以选
                mask.append(1.0)

        return np.array(mask, dtype=np.float32), has_urgent

    def agent_observe(self, cid):
        i = self.cid2ind[cid]
        obs = self.agents[i].observe_space
        return obs

    def order_agents(self):
        agents_value = [(agent.id, agent.value) for agent in self.agents]
        agents_order = sorted(agents_value, key=lambda k:k[1])
        # self.agents_order = [(cid, i) for i, (cid, _) in enumerate(agents_order)]
        self.agent_dict = {cid: i for i, (cid, _) in enumerate(agents_order)}
        return [(cid, i) if value!=0 else (cid, value) for i, (cid, value) in enumerate(agents_order)]

    def is_feasible(self, cid, action):
        room_option_ind, time_option_ind, penalty = action
        self.Hard_validator.setClasses(self.agents)
        confilct_agent_cid = set()
        if room_option_ind != -1:
            i = self.cid2ind[cid]
            room_option = self.agents[i].room_options[room_option_ind]
            # TODO
            # 1. 同一间 room 在同一 time 不能被两个 class 占用
            if self.rooms[room_option['id']]['ocupied']:
                violate = self.Hard_validator.RoomConflicts(cid, self.rooms[room_option['id']]['ocupied'])
                self.check_agent(cid, ("RoomConflicts", violate))
                if violate: return False, []
            # ...
            # 2. room 自身的 unavailable 时间不能选
            violate = self.Hard_validator.RoomUnavailable(cid, self.rooms[room_option['id']]['unavailables_bits'])
            self.check_agent(cid, ("RoomUnavailable", violate), rid=room_option['id'])
            if violate: return False, []
        # 3. hard_constraints
        for hard_constraint in self.hard_constrains:
            if cid in hard_constraint['classes']:
                violate = self.Hard_validator._violation_rate(hard_constraint, cid)
                if violate: 
                    self.check_agent(cid, (f"HardConstraint {hard_constraint}", violate))
                    return False, []
                else:
                    for ccid in hard_constraint['classes']:
                        if cid != ccid:
                            ci = self.cid2ind[ccid]
                            if self.agents[ci].action==None:
                                confilct_agent_cid.add(ccid)
        return True, list(confilct_agent_cid)

    def incremental_penalty(self, cid, action):
        p = action[2]
        self.Soft_validator.setClasses(self.agents)
        for soft_constrain in self.soft_constrains:
            # TODO
            if cid in soft_constrain['classes']:
                violation_rate = self.Soft_validator._violation_rate(soft_constrain, cid)
                if violation_rate:
                    p += violation_rate * soft_constrain['penalty']
        return p

    def apply_action(self, cid, chosen_action, chosen_penalty):
        i = self.cid2ind[cid]
        room_option_ind, time_option_ind, penalty = chosen_action
        self.agents[i].candidate = None
        self.agents[i].action = chosen_action
        self.agents[i].penalty = chosen_penalty
        time_option = self.agents[i].time_options[time_option_ind]
        # self.timeTable_matrix = np.add(self.timeTable_matrix, time_option['optional_time'])
        if room_option_ind != -1:
            room_option = self.agents[i].room_options[room_option_ind]
            self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], penalty)) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}
         

    def total_penalty(self):
        penalty = 0
        Room_penalty = 0
        Time_penalty = 0
        for agent in self.agents:
            if agent.id not in self._assignment:
                penalty += agent.penalty
                rid, tid, _ = agent.action
                if agent.room_required:
                    Room_penalty += agent.room_options[rid]['penalty']
                Time_penalty += agent.time_options[tid]['penalty']
        self.Soft_validator.setClasses(self.agents)
        Distribution_penalty = 0
        for soft_constrain in self.soft_constrains:
            # TODO
            violation_rate = self.Soft_validator._violation_rate(soft_constrain)
            if violation_rate:
                Distribution_penalty += violation_rate * soft_constrain['penalty']
                penalty += violation_rate * soft_constrain['penalty']
        return {
            "not assignment": self._assignment,
            "penalty": penalty,
            "Total_cost": "TODO",
            "Student conflicts": "TODO",
            "Time penalty": Time_penalty,
            "Room penalty": Room_penalty,
            "Distribution penalty": Distribution_penalty,
        }

    def agent_value_update(self, conflicts):
        for ccid in conflicts:
            i = self.cid2ind[ccid]
            agent = self.agents[i]
        # for agent in self.agents:
            if agent.id not in self._sched:
                action_space = []
                for aid, action in enumerate(agent.action_space):
                    agent.candidate = action
                    feasible, conflicts = self.is_feasible(agent.id, action)
                    if not feasible:
                        continue
                    else:
                        action_space.append(action)
                agent.action_space = action_space
                agent.value = len(action_space)

    def step_course(self, cid, action_idx):
        i = self.cid2ind[cid]
        agent = self.agents[i]

        valid_actions = []
        best_penalty = float("inf")

        # 原来在 step() 里对这一门课枚举 action_space 的那段逻辑抽出来
        for aid, action in enumerate(agent.action_space):
            agent.candidate = action
            feasible, conflicts = self.is_feasible(cid, action)
            if not feasible:
                continue
            penalty = self.incremental_penalty(cid, action)
            valid_actions.append((aid, penalty))
            if penalty < best_penalty:
                best_penalty = penalty

        # === 关键分支：没有合法 action ===
        if len(valid_actions) == 0:
            # 标记无合法动作
            no_valid = True

            # 你可以定义一个比较大的惩罚，比如：
            # invalid_penalty = dead_penalty
            reward = 1 / dead_reward

            # 这门课是否标记为 done
            self._assignment.append(cid)
            self._sched.append(cid)
            done_course = True

            # 可以给它一个特殊的观察（比如全 -1），方便调试
            next_course_obs = -np.ones_like(agent.observe_space, dtype=np.float64)

            # Scheduler 的下一个全局状态（这一步可以直接复用当前状态，或者让 env 内部记录“失败标志”）
            next_sched_state = self.get_scheduler_obs()

            return reward, next_sched_state, next_course_obs, done_course, no_valid, conflicts

        # === 正常情况：有合法动作 ===
        else:
            no_valid = False

            # 按照你原来的逻辑，从 valid_actions 里选一个
            action_idx = action_idx % len(valid_actions)
            chosen_aid, chosen_penalty = valid_actions[action_idx]
            chosen_action = agent.action_space[chosen_aid]

            self.apply_action(cid, chosen_action, chosen_penalty)
            reward = 1 / chosen_penalty if chosen_penalty else 1
            done_course = True
            self._sched.append(cid)

            next_course_obs = self.agent_observe(cid)
            next_sched_state = self.get_scheduler_obs()

            return reward, next_sched_state, next_course_obs, done_course, no_valid, conflicts

    def check(self, type_name="SameRoom"):
        for constraint in self.hard_constrains:
            if constraint['type'] == type_name:
                print(type_name)
                if type_name in ["SameRoom", "DifferentRoom"]:
                    for cid in constraint['classes']:
                        i = self.cid2ind[cid]
                        print(f"agent {cid}: Room {self.agents[i].action[0]}", end=" | ")
                if type_name in ["SameStart", "SameTime", "DifferentTime", "Precedence"]:
                    for cid in constraint['classes']:
                        i = self.cid2ind[cid]
                        if self.agents[i].action:
                            oid = self.agents[i].action[1]
                            print(f"agent {cid}: time {self.agents[i].time_options[oid]['optional_time_bits']}", end=" | ")
                        else:
                            print(f"agent {cid}: time not assign", end=" | ")
                print("")

    def results(self):
        assignment = {}
        for agent in self.agents:
            result = agent.result()
            assignment.update(result)
        return assignment
    
    ############################################################################
    # test
    ############################################################################
    def get_agent_values(self, ids):
        values = []
        if type(ids[0]) == str:
            cids = ids
        else:
            cids = [self._agents[ind] for ind in ids]
        for cid in cids:
            i = self.cid2ind[cid]
            values.append((cid, self.agents[i].value))
        return values
    
    def check_agent(self, cid, s, rid=None):
        if cid == "-1":
            i = self.cid2ind[cid]
            if s[1] == True and s[0] not in ["RoomConflicts", "RoomUnavailable"]:
                print(f"agent {cid} {s}: ")
                print(f"  candidate: {self.agents[i].candidate}")
                print(f"  room options: {self.agents[i].room_options[self.agents[i].candidate[0]] if self.agents[i].candidate[0]!=-1 else 'N/A'}")
                print(f"  time options: {self.agents[i].time_options[self.agents[i].candidate[1]]['optional_time_bits']}")
        if cid == "-1":
            i = self.cid2ind[cid]
            print(f"agent {cid} {s}: ")
            if self.agents[i].room_options[self.agents[i].candidate[0]]['id'] == '3':
                print(f"  candidate: {self.agents[i].candidate}")
                print(f"  room options: {self.agents[i].room_options[self.agents[i].candidate[0]] if self.agents[i].candidate[0]!=-1 else 'N/A'}")
                print(f"  time options: {self.agents[i].time_options[self.agents[i].candidate[1]]['optional_time_bits']}")
                print(f"  action: {self.agents[i].action}")
                print(f"  penalty: {self.agents[i].penalty}")
                # print(f"  value: {self.agents[i].value}")
            print("")
        # if rid == '3':
        #     if  cid == "246":
        #         i = self.cid2ind[cid]
        #         print(f"agent {cid} Room {rid} {s}: ")
        #         print(f"  candidate: {self.agents[i].candidate}")
        #         print(f"  room options: {self.agents[i].room_options[self.agents[i].candidate[0]] if self.agents[i].candidate[0]!=-1 else 'N/A'}")
        #         print(f"  time options: {self.agents[i].time_options[self.agents[i].candidate[1]]['optional_time_bits']}")
        #         print(f"  action: {self.agents[i].action}")
        #         print(f"  penalty: {self.agents[i].penalty}")
        #         print(f"  value: {self.agents[i].value}")
            # print(cid, end=" ")
    
    def save(self, filename):
        print("model.save", self.total_penalty()["penalty"])
        data = {
            "classes":{agent.id: agent.action for agent in self.agents},
            "rooms": self.rooms
        }
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False)