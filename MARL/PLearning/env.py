import copy
import json
import numpy as np
from math import inf
from MARL.PLearning.agents import agent_class
from utils.constraints import HardConstraints, SoftConstraints

class CustomEnvironment:
    def __init__(self, reader, config=None):
        self.reader = reader
        self.w1 = config["train"]["agent_rewards"]["weight1"]
        self.w2 = config["train"]["agent_rewards"]["weight2"]
        self.w3 = config["train"]["agent_rewards"]["weight3"]

        self.optimization = self.reader.optimization
        
        self.hard_constrains = self.reader.distributions['hard_constraints']
        self.soft_constrains = self.reader.distributions['soft_constraints']
        self.Hard_validator = HardConstraints()
        self.Soft_validator = SoftConstraints()

        self.agents = [] # classes
        self.cid2ind = {}
        for i, (key, each) in enumerate(self.reader.classes.items()):
            self.agents.append(agent_class(each, len(self.reader.classes), self.optimization))
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

    def reset(self):
        self._assignment = []
        self.rooms = copy.deepcopy(self.reader.rooms)
        self.order_agents() # (cid, value)
        agent_observations = {}
        scheduler_observations = []
        for agent in self.agents:
            agent.value = len(agent.action_space)
            agent.candidate = None
            agent.action = None
            agent.penalty = inf
            agent.choice = 0
            agent.masked_actions = [1 for _ in range(len(agent.action_space))]
            agent.observe_space = [0 for action in agent.action_space]
            agent_observations[agent.id] = self.agent_observe(agent.id)
        none_assignment = [agent.id for agent in self.agents]
        scheduler_observations = [self.agent_order_dict[agent.id] for agent in self.agents]
        return agent_observations, scheduler_observations, none_assignment
    
    def order_agents(self):
        agents_id = [agent.id for agent in self.agents]
        agents_value = zip(agents_id, self.agents_value.tolist())
        agents_order = sorted(agents_value, key=lambda k:k[1])
        self.agent_order_dict = {cid: i for i, (cid, _) in enumerate(agents_order)}
        self.agents_order = {i: cid for i, (cid, _) in enumerate(agents_order)}

    def reset_step(self, sched_obs, sched_mask, actions):
        self.apply_scheduling(sched_obs, sched_mask, actions)
        self._assignment = []
        self.rooms = copy.deepcopy(self.reader.rooms)
        observations = {}
        for agent in self.agents:
            agent.candidate = None
            agent.action = None
            agent.penalty = inf
            agent.masked_actions = np.array([1 for _ in range(len(agent.action_space))], dtype=np.int8)
            agent.observe_space = [0 for action in agent.action_space]
            observations[agent.id] = self.agent_observe(agent.id)
        return observations
    
    def apply_scheduling(self, states, masks, actions):
        it = 0
        for oi1, (ind1, mask) in enumerate(zip(states, masks)):
            if mask == 0:
                it += 1
                oi2 = actions[oi1]
                cid2 = self.agents_order[oi2]
                cid1 = self.agents[oi1].id
                # if it <= 5:
                #     print(f"oi2 {oi2}")
                #     print(f"Cid1 {cid1}<-->Cid2 {cid2}")
                #     print(f"order1 {self.agent_order_dict[cid1]}<-->order2 {self.agent_order_dict[cid2]}")
                i1, i2 = self.agent_order_dict[cid1], self.agent_order_dict[cid2]
                self.agents_order[i1], self.agents_order[i2] = self.agents_order[i2], self.agents_order[i1]
                self.agent_order_dict[cid1], self.agent_order_dict[cid2] = self.agent_order_dict[cid2], self.agent_order_dict[cid1]

    def agent_observe(self, cid):
        i = self.cid2ind[cid]
        obs = self.agents[i].observe_space
        return obs

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
                    return False
            # ...
            # 2. room 自身的 unavailable 时间不能选
            violate = self.Hard_validator.RoomUnavailable(cid, self.rooms[room_option['id']]['unavailables_bits'])
            if violate: 
                return False
        # 3. hard_constraints
        for hard_constraint in self.hard_constrains:
            if cid in hard_constraint['classes']:
                violate = self.Hard_validator._violation_rate(hard_constraint, cid)
                if violate: return False
        return True

    def incremental_penalty(self, cid, action):
        p = action[2]
        self.Soft_validator.setClasses(self.agents)
        for soft_constrain in self.soft_constrains:
            if cid in soft_constrain['classes']:
                violation_rate = self.Soft_validator._violation_rate(soft_constrain, cid)
                if violation_rate:
                    p += violation_rate * soft_constrain['penalty']
        return p
    
    def handle_infeasible_case(self, cid):
        self._assignment.append(cid)

    def apply_action(self, cid):
        i = self.cid2ind[cid]
        observe = self.agents[i].observe_space
        observe = np.max(observe) - observe + self.agents[i].masked_actions
        observe = np.array(observe, dtype=np.float64)
        observe = observe*self.agents[i].masked_actions
        observe = observe/np.sum(observe)
        action_ind = self.agents[i]._action_space.sample(probability=observe)
        action = self.agents[i].action_space[action_ind]
        self.agents[i].observe_space = observe.tolist()
        room_option_ind, time_option_ind, penalty = action
        self.agents[i].candidate = None
        self.agents[i].action = action
        self.agents[i].penalty = penalty
        time_option = self.agents[i].time_options[time_option_ind]
        if room_option_ind != -1:
            room_option = self.agents[i].room_options[room_option_ind]
            self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], action)) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}
        
    def total_penalty(self):
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
                # print(f"cid {cid} ind {ind} none_assignment")
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
        for soft_constrain in self.soft_constrains:
            violation_rate = self.Soft_validator._violation_rate(soft_constrain)
            if violation_rate:
                Distribution_penalty += violation_rate * soft_constrain['penalty']
                penalty += violation_rate * soft_constrain['penalty']
        Total_cost = self.optimization["time"] * Time_penalty + \
                        self.optimization["room"] * Room_penalty + \
                        self.optimization["distribution"] * Distribution_penalty + \
                        self.optimization["student"] * Student_penalty
        rewards = []
        r1 = -len(self._assignment) / len(self.agents)
        for agent in self.agents:
            if agent.id in self._assignment:
                r2 = -agent.max_penalty
            else:
                r2 = agent.max_penalty - Total_cost / len(self.agents)
            r3 = -agent.long_term_penalty
            reward = self.w1 * r1 + self.w2 * r2 + self.w3 * r3
            rewards.append(reward)
        return {
            "not assignment": self._assignment,
            "penalty": penalty,
            "rewards": rewards,
            "Total cost": Total_cost,
            "Time penalty": Time_penalty,
            "Room penalty": Room_penalty,
            "Distribution penalty": Distribution_penalty,
            # "observations": observations,
            "scheduler_observations": scheduler_observations,
            "scheduler_mask": scheduler_mask
        }

    def step(self):
        for _, cid in self.agents_order.items():
            i = self.cid2ind[cid]
            valid = False
            for aid, action in enumerate(self.agents[i].action_space):
                self.agents[i].candidate = action
                if not self.is_feasible(cid, action):
                    self.agents[i].observe_space[aid] = 0
                    self.agents[i].masked_actions[aid] = 0
                    continue
                
                valid = True
                p = self.incremental_penalty(cid, action)
                self.agents[i].observe_space[aid] = p
                self.agents[i].masked_actions[aid] = 1
            if not valid:
                self.handle_infeasible_case(cid)
                continue
                # break
            self.apply_action(cid)
        # self.check("Precedence")
        return self.total_penalty()

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
            "rooms": self.rooms
        }
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False)