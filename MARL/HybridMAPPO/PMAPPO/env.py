import copy
import json
import numpy as np
from math import inf
from MARL.HybridMAPPO.PMAPPO.agents import agent_class
from utils.constraints import HardConstraints, SoftConstraints
from utils.solutionReader import PSTTReader

class CustomEnvironment:
    def __init__(self, reader, config=None):
        self.reader = reader
        self.insist = config['train']['insist']
        self.optimization = self.reader.optimization
        
        self.hard_constraints = self.reader.distributions['hard_constraints']
        self.soft_constraints = self.reader.distributions['soft_constraints']
        self.Hard_validator = HardConstraints()
        self.Soft_validator = SoftConstraints()

        self.agents = [] # classes
        self.cid2ind = {}
        self.roomRelatedClass = {}
        for i, (key, each) in enumerate(self.reader.classes.items()):
            agent = agent_class(each, len(self.reader.classes), self.optimization)
            self.agents.append(agent)
            if self.roomRelatedClass.get(key, 0) == 0: self.roomRelatedClass[key] = set()
            self.roomRelatedClass[key].update(agent.rooms)
            self.cid2ind[key] = i
        
        self.travel = self.reader.travel
        self.Hard_validator.setTravel(self.travel)
        self.Hard_validator.sefnrDays(self.reader.nrDays)
        self.Hard_validator.sefnrWeeks(self.reader.nrWeeks)
        self.Hard_validator.setCid2ind(self.cid2ind)
        self.Soft_validator.setTravel(self.travel)
        self.Soft_validator.sefnrDays(self.reader.nrDays)
        self.Soft_validator.sefnrWeeks(self.reader.nrWeeks)
        self.Soft_validator.setCid2ind(self.cid2ind)
        self.none_assignment = [agent.id for agent in self.agents]

        self.rooms = copy.deepcopy(self.reader.rooms) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}
        self.agents_value = np.array([agent.value for agent in self.agents])
        self.max_value = np.max(self.agents_value)
        self.penalty_benchmark = inf
        self.getAgentConstraintSets()

    def getAgentConstraintSets(self):
        for hard_constraint in self.hard_constraints:
            for ccid1 in hard_constraint['classes']:
                ind1 = self.cid2ind[ccid1]
                self.agents[ind1].hard_constraints += 1
                for ccid2 in hard_constraint['classes']:
                    if ccid1 != ccid2:
                        ind2 = self.cid2ind[ccid2]
                        self.agents[ind1].hard_constraints_cids.add(ccid2)
                        self.agents[ind2].hard_constraints_cids.add(ccid1)
        for soft_constraint in self.soft_constraints:
            for ccid1 in soft_constraint['classes']:
                ind1 = self.cid2ind[ccid1]
                self.agents[ind1].soft_constraints += 1
                for ccid2 in soft_constraint['classes']:
                    if ccid1 != ccid2:
                        ind2 = self.cid2ind[ccid2]
                        self.agents[ind1].soft_constraints_cids.add(ccid2)
                        self.agents[ind2].soft_constraints_cids.add(ccid1)

    def getRoomConflicts(self, cid):
        i = self.cid2ind[cid]
        for action in self.agents[i].action_space:
            room_option_ind, time_option_ind, penalty = action
            room_option = None
            self.Hard_validator.setClasses(self.agents)
            if room_option_ind != -1:
                room_option = self.agents[i].room_options[room_option_ind]
                if self.rooms[room_option['id']]['ocupied']:
                    violate = self.Hard_validator.RoomConflicts(cid, self.rooms[room_option['id']]['ocupied'])
                    if violate: 
                        for ccid in self.roomRelatedClass[room_option['id']]:
                            if cid != ccid:
                                ci = self.cid2ind[ccid]
                                if self.agents[ci].action!=None:
                                    self.agents[i].room_constraints_cids.add(ccid)
                violate = self.Hard_validator.RoomUnavailable(cid, self.rooms[room_option['id']]['unavailables_bits'])
                if violate: 
                    for ccid in self.roomRelatedClass[room_option['id']]:
                            if cid != ccid:
                                ci = self.cid2ind[ccid]
                                if self.agents[ci].action!=None:
                                    self.agents[i].room_constraints_cids.add(ccid)

    def initial_solution(self, solution_path, value_path):
        solution = PSTTReader(solution_path)
        Room_penalty = 0
        Time_penalty = 0
        Distribution_penalty = 0
        Student_penalty = 0
        self._assignment = []
        scheduler_observations = []
        actions = {}
        masks = {}
        if value_path:
            with open(value_path, 'r') as f:
                values = json.load(f)
                max_value = max(values.values())
                self.agents_value = np.array([values.get(agent.id, max_value) for agent in self.agents])
                self.order_agents()
                scheduler_observations = [self.agent_order_dict[agent.id] for agent in self.agents]
        for i, (cid, solu) in enumerate(solution.classes.items()):
            ind = self.cid2ind[cid]
            agent = self.agents[ind]
            optional_time = solu['optional_time']
            if optional_time:
                weeks_bits, days_bits, start = optional_time
            else:
                self._assignment.append(cid)
                continue
            rid = solu['room']
            if agent.room_required and rid == None:
                self._assignment.append(cid)
                continue
            action_idx = agent.get_action_idx(weeks_bits, days_bits, start, rid)
            actions[cid] = action_idx
            mask = np.zeros(shape=self.max_value, dtype=np.int8)
            agent.mask = np.array([0 for _ in range(len(agent.action_space))], dtype=np.int8)
            mask[action_idx] = 1
            masks[cid] = mask
            agent.isfixed = True
            agent.mask[action_idx] = 1
            action = agent.action_space[action_idx]
            room_option_ind, time_option_ind, penalty = action
            agent.action = action
            agent.last_action = action
            agent.penalty = penalty
            time_option = agent.time_options[time_option_ind]
            if room_option_ind != -1:
                room_option = agent.room_options[room_option_ind]
                self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], action))
            if agent.room_required:
                Room_penalty += room_option['penalty']
            Time_penalty += time_option['penalty']
        self.Soft_validator.setClasses(self.agents)
        for soft_constraint in self.soft_constraints:
            violation_rate = self.Soft_validator._violation_rate(soft_constraint)
            if violation_rate:
                Distribution_penalty += violation_rate * soft_constraint['penalty']
        Total_cost = self.optimization["time"] * Time_penalty + \
                        self.optimization["room"] * Room_penalty + \
                        self.optimization["distribution"] * Distribution_penalty + \
                        self.optimization["student"] * Student_penalty
        quality = {
            "not assignment": self._assignment,
            "Total cost": Total_cost,
            "Time penalty": Time_penalty,
            "Room penalty": Room_penalty,
            "Distribution penalty": Distribution_penalty,
        }
        self.penalty_benchmark = Total_cost
        mappo_observations, _ = self.agents_observe()
        self.rooms = copy.deepcopy(self.reader.rooms)
        for agent in self.agents:
            agent.candidate = None
            agent.action = None
            agent.penalty = inf
        return mappo_observations, scheduler_observations, masks, actions, quality
    
    def reset(self, order=False):
        self._assignment = []
        self.rooms = copy.deepcopy(self.reader.rooms)
        if order: self.order_agents() # (cid, value)
        scheduler_observations = []
        for agent in self.agents:
            agent.value = len(agent.action_space)
            agent.candidate = None
            agent.action = None
            agent.penalty = inf
            agent.choice = 0
            agent.masked_actions = np.array([1 for _ in range(len(agent.action_space))], dtype=np.int8)
            agent.observe_space = np.array([0 for _ in range(len(self.agents))], dtype=np.float64)
            agent.room_constraints_cids = set()
        mappo_observations, masks = self.agents_observe()
        none_assignment = [agent.id for agent in self.agents]
        scheduler_observations = [self.agent_order_dict[agent.id] for agent in self.agents]
        return mappo_observations, masks, scheduler_observations, none_assignment
    
    def order_agents(self):
        agents_id = [agent.id for agent in self.agents]
        agents_value = zip(agents_id, self.agents_value.tolist())
        agents_order = sorted(agents_value, key=lambda k:k[1])
        self.agent_order_dict = {cid: i for i, (cid, _) in enumerate(agents_order)}
        self.agents_order = {i: cid for i, (cid, _) in enumerate(agents_order)}

    def reset_step(self):
        # self.apply_scheduling(sched_obs, sched_mask, actions)
        self._assignment = []
        masks = {}
        self.rooms = copy.deepcopy(self.reader.rooms)
        for agent in self.agents:
            mask = np.zeros(shape=self.max_value, dtype=np.int8)
            agent.mask = np.array([0 for _ in range(len(agent.action_space))], dtype=np.int8)
            if agent.action:
                choice = np.random.choice([0, 1], p=[1 - self.insist, self.insist])
                if choice:
                    # 75%保留之前的解
                    aidx = agent.action_space.index(agent.last_action)
                    mask[aidx] = 1
                    agent.isfixed = True
                    agent.mask[aidx] = 1
                else:
                    # 25%不能选之前的解
                    aidx = agent.action_space.index(agent.last_action)
                    mask[:agent.value] = 1
                    mask[aidx] = 0
                    agent.isfixed = False
                    agent.mask[:] = 1
                    agent.mask[aidx] = 0
            else:
                mask[:agent.value] = 1
                agent.mask[:] = 1
                agent.isfixed = False
            masks[agent.id] = mask
        for agent in self.agents:
            agent.candidate = None
            agent.action = None
            agent.penalty = inf
            agent.masked_actions = np.array([1 for _ in range(len(agent.action_space))], dtype=np.int8)
            agent.observe_space = np.array([0 for _ in range(len(self.agents))], dtype=np.float64)
            agent.room_constraints_cids = set()
        return masks
    
    def apply_scheduling(self, states, masks, actions):
        for oi1, (ind1, mask) in enumerate(zip(states, masks)):
            if mask == 0:
                oi2 = actions[oi1]
                cid2 = self.agents_order[oi2]
                cid1 = self.agents[oi1].id
                i1, i2 = self.agent_order_dict[cid1], self.agent_order_dict[cid2]
                self.agents_order[i1], self.agents_order[i2] = self.agents_order[i2], self.agents_order[i1]
                self.agent_order_dict[cid1], self.agent_order_dict[cid2] = self.agent_order_dict[cid2], self.agent_order_dict[cid1]

    def agents_observe(self):
        observations = {}
        masks = {}
        for agent in self.agents:
            observations[agent.id] = self.agent_observe(agent.id)
            masks[agent.id] = self.agent_mask(agent.id)
        return observations, masks

    def agent_observe(self, cid1):
        ind1 = self.cid2ind[cid1]
        oi = self.agent_order_dict[cid1]
        for oi2 in range(oi+1, len(self.agents)):
            cid2 = self.agents_order[oi2]
            ind2 = self.cid2ind[cid2]
            if self.agents[ind2].id in self.agents[ind1].hard_constraints_cids:
                self.agents[ind1].observe_space[oi2] = 1
            elif self.agents[ind2].id in self.agents[ind1].room_constraints_cids:
                self.agents[ind1].observe_space[oi2] = 0.5
        return self.agents[ind1].observe_space

    def agent_mask(self, cid):
        ind = self.cid2ind[cid]
        mask = np.zeros(shape=self.max_value, dtype=np.int8)
        mask[:self.agents[ind].value] = 1
        return mask

    def is_feasible(self, cid, action):
        room_option_ind, time_option_ind, penalty = action
        self.Hard_validator.setClasses(self.agents)
        if room_option_ind != -1:
            i = self.cid2ind[cid]
            room_option = self.agents[i].room_options[room_option_ind]
            # 1. 同一间 room 在同一 time 不能被两个 class 占用
            if self.rooms[room_option['id']]['ocupied']:
                violate = self.Hard_validator.RoomConflicts(cid, self.rooms[room_option['id']]['ocupied'])
                if violate: 
                    return False, "RoomConflicts"
            # ...
            # 2. room 自身的 unavailable 时间不能选
            violate = self.Hard_validator.RoomUnavailable(cid, self.rooms[room_option['id']]['unavailables_bits'])
            if violate: 
                return False, "RoomUnavailable"
        # 3. hard_constraints
        for hard_constraint in self.hard_constraints:
            if cid in hard_constraint['classes']:
                violate = self.Hard_validator._violation_rate(hard_constraint, cid)
                if violate: return False, hard_constraint["type"]
        return True, None

    def incremental_penalty(self, cid, action):
        p = action[2]
        soft_penalty = 0
        self.Soft_validator.setClasses(self.agents)
        for soft_constrain in self.soft_constraints:
            if cid in soft_constrain['classes']:
                violation_rate = self.Soft_validator._violation_rate(soft_constrain, cid)
                if violation_rate:
                    soft_penalty += violation_rate * soft_constrain['penalty']
        p +=  soft_penalty * self.optimization["distribution"]
        return p
    
    def handle_infeasible_case(self, cid):
        self._assignment.append(cid)
        self.getRoomConflicts(cid)

    def apply_mappo_action(self, probs):
        for i, agent in enumerate(self.agents):
            agent.probs = probs[i][:agent.value]

    def apply_action(self, cid):
        i = self.cid2ind[cid]
        agent = self.agents[i]
        if agent.isfixed:
            agent.masked_actions = agent.mask * agent.masked_actions
            action_ind = agent.action_space.index(agent.last_action)
        else:
            probs = (agent.probs + 1e-6) * agent.masked_actions
            probs = np.array(probs, dtype=np.float64)
            probs = probs/np.sum(probs)
            action_ind = self.agents[i]._action_space.sample(probability=probs)
        action = self.agents[i].action_space[action_ind]
        room_option_ind, time_option_ind, penalty = action
        self.agents[i].candidate = None
        self.agents[i].action = action
        self.agents[i].penalty = penalty
        time_option = self.agents[i].time_options[time_option_ind]
        if room_option_ind != -1:
            room_option = self.agents[i].room_options[room_option_ind]
            self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], action)) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}
        return action_ind, agent.masked_actions

    def total_penalty(self, actions=None, masked_actions=None):
        penalty = 0
        Room_penalty = 0
        Time_penalty = 0
        Student_penalty = 0
        # observations = {}
        scheduler_observations = []
        scheduler_mask = []
        for agent in self.agents:
            cid = agent.id
            scheduler_observations.append(self.agent_order_dict[cid])
            ind = self.cid2ind[cid]
            if cid in self._assignment:
                scheduler_mask.append(0)
            else:
                scheduler_mask.append(1)
        for agent in self.agents:
            # if agent.id in self._assignment:
            #     observations[agent.id] =self.agent_observe(agent.id)
            if agent.id not in self._assignment:
                penalty += agent.penalty
                rid, tid, _ = agent.action
                if agent.room_required:
                    Room_penalty += agent.room_options[rid]['penalty']
                Time_penalty += agent.time_options[tid]['penalty']
        self.Soft_validator.setClasses(self.agents)
        Distribution_penalty = 0
        for soft_constrain in self.soft_constraints:
            violation_rate = self.Soft_validator._violation_rate(soft_constrain)
            if violation_rate:
                Distribution_penalty += violation_rate * soft_constrain['penalty']
                penalty += violation_rate * soft_constrain['penalty']
        Total_cost = self.optimization["time"] * Time_penalty + \
                        self.optimization["room"] * Room_penalty + \
                        self.optimization["distribution"] * Distribution_penalty + \
                        self.optimization["student"] * Student_penalty
        rewards = []
        #TODO
        r1 = len(self._assignment)
        for agent in self.agents:
            if agent.id in self._assignment: r2 = 0
            else:
                if agent.action == agent.last_action:
                    r2 = np.max(agent.action_penalty) - r1 * Total_cost / len(self.agents)
                else:
                    r2 = np.max(agent.action_penalty)
            if len(self._assignment) == 0:
                agent.last_action = agent.action
            r3 = (self.penalty_benchmark - Total_cost) / len(self.agents)
            reward = r2 + r3
            rewards.append(reward)
        mappo_observations, _ = self.agents_observe()
        return {
            "not assignment": self._assignment,
            "penalty": penalty,
            "rewards": rewards,
            "Total cost": Total_cost,
            "Time penalty": Time_penalty,
            "Room penalty": Room_penalty,
            "Distribution penalty": Distribution_penalty,
            "mappo_observations": mappo_observations,
            "scheduler_observations": scheduler_observations,
            "scheduler_mask": scheduler_mask,
            "actions": actions,
            "masked_actions": masked_actions
        }

    def step(self):
        actions = {}
        masked_actions = {}
        count = 0
        for _, cid in self.agents_order.items():
            i = self.cid2ind[cid]
            valid = False
            for aid, action in enumerate(self.agents[i].action_space):
                self.agents[i].candidate = action
                flag, name = self.is_feasible(cid, action)
                if not flag:
                    if self.agents[i].action == self.agents[i].last_action:
                        print(f"agent {cid} action {aid} no longer a valid action: {name}")
                    self.agents[i].action_penalty[aid] = 0
                    self.agents[i].masked_actions[aid] = 0
                    continue
                
                valid = True
                p = self.incremental_penalty(cid, action)
                self.agents[i].action_penalty[aid] = p
                self.agents[i].masked_actions[aid] = 1
            if not valid:
                self.handle_infeasible_case(cid)
                actions[cid] = 0
                masked_actions[cid] = self.agents[i].masked_actions
                continue
                # break
            action_ind, mask = self.apply_action(cid)
            if self.agents[i].action != self.agents[i].last_action:
                count += 1
            actions[cid] = action_ind
            masked_actions[cid] = mask
        print(f"{count}/{len(self.agents)} agents choose differently. {len(self._assignment)}/{len(self.agents)} agents are not assigned.")
        # self.check("Precedence")
        return self.total_penalty(actions, masked_actions)

    def results(self):
        assignment = {}
        for agent in self.agents:
            result = agent.result()
            assignment.update(result)
        return assignment
    
    ############################################################################
    # tools
    ############################################################################
    def get_agent_values(self, ids):
        values = []
        for cid in ids:
            i = self.cid2ind[cid]
            # values.append((cid, self.agents[i].value))
        return values
    
    def save(self, filename):
        data = {
            "classes":{agent.id: agent.action for agent in self.agents},
        }
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False)