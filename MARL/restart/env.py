import copy
import random
import numpy as np
import json
from MARL.restart.agents import agent_class
from MARL.restart.constraints import HardConstraints, SoftConstraints
from math import inf

class CustomEnvironment:
    def __init__(self, reader, action_shape=32, weight_a=100, weight_p=1, weight_lt=0.5):
        self.reader = reader
        self.action_shape = action_shape
        self.obs_shape = len(self.reader.classes)

        self.hard_constrains = self.reader.distributions['hard_constraints']
        self.soft_constrains = self.reader.distributions['soft_constraints']
        self.Hard_validator = HardConstraints()
        self.Soft_validator = SoftConstraints()

        self.agents = [] # classes
        self.cid2ind = {}
        self.roomRelatedClass = {}
        i = 0
        for key, each in self.reader.classes.items():
            agent = agent_class(each, self.obs_shape)
            self.agents.append(agent)
            if self.roomRelatedClass.get(key, 0) == 0: self.roomRelatedClass[key] = set()
            self.roomRelatedClass[key].update(agent.rooms)
            self.cid2ind[key] = i
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
        self.not_assignment = [agent.id for agent in self.agents]

        self.rooms = copy.deepcopy(self.reader.rooms) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}

        self.getAgentConstraintSets()
        self.weight_a = len(self.reader.classes)//2
        self.weight_p = weight_p
        self.weight_lt = weight_lt
        self.agents_value = np.array([agent.value for agent in self.agents])

    def getAgentConstraintSets(self):
        for hard_constraint in self.hard_constrains:
            for ccid1 in hard_constraint['classes']:
                ind1 = self.cid2ind[ccid1]
                self.agents[ind1].hard_constraints += 1
                for ccid2 in hard_constraint['classes']:
                    if ccid1 != ccid2:
                        ind2 = self.cid2ind[ccid2]
                        self.agents[ind1].hard_constraints_cids.add(ccid2)
                        self.agents[ind2].hard_constraints_cids.add(ccid1)
        for soft_constraint in self.soft_constrains:
            for ccid1 in soft_constraint['classes']:
                ind1 = self.cid2ind[ccid1]
                self.agents[ind1].soft_constraints += 1
                for ccid2 in soft_constraint['classes']:
                    if ccid1 != ccid2:
                        ind2 = self.cid2ind[ccid2]
                        self.agents[ind1].soft_constraints_cids.add(ccid2)
                        self.agents[ind2].soft_constraints_cids.add(ccid1)

    def order_agents(self):
        agents_id = [agent.id for agent in self.agents]
        # self.agents_value -= np.max(self.agents_value)
        # self.agents_value = np.exp(self.agents_value) / np.sum(self.agents_value)
        agents_value = zip(agents_id, self.agents_value.tolist())
        agents_order = sorted(agents_value, key=lambda k:k[1])
        self.agent_order_dict = {cid: i for i, (cid, _) in enumerate(agents_order)}
        self.agents_order = {i: (cid, value) for i, (cid, value) in enumerate(agents_order)}

    def reset(self):
        self.rooms = copy.deepcopy(self.reader.rooms)
        self.order_agents() # (cid, value)
        observations = {}
        for agent in self.agents:
            agent.value = len(agent.action_space)
            self.candidate = None
            agent.action = None
            agent.penalty = agent.avg_penalty
            agent.best_penalty = agent.avg_penalty
            agent.long_term_penalty = 0
            agent.choice = 0
            observations[agent.id] = self.agent_observe(agent.id)
        self.not_assignment  = []
        dones = [agent.id for agent in self.agents]
        return observations, dones
    
    def action_not_assignment(self, cid, action):
        i = self.cid2ind[cid]
        # order = self.agent_order_dict[cid]
        if action:
            self.agents_value[i] *= action/10

    def action_assignment(self, cid, action):
        i = self.cid2ind[cid]
        self.agents[i].choice = action

    def apply_sched_action(self, action_dict):
        for i, (cid, value) in enumerate(self.agents_order.values()):
            if cid in self.not_assignment:
                self.action_not_assignment(cid, action_dict[cid])

    def apply_action(self, action_dict):
        for i, (cid, value) in enumerate(self.agents_order.values()):
            # if cid in self.not_assignment:
            #     self.action_not_assignment(cid, action_dict[cid])
            # else:
            self.action_assignment(cid, action_dict[cid])

    def step_reset(self):
        self.rooms = copy.deepcopy(self.reader.rooms)
        # self.order_agents() # (cid, value)
        for agent in self.agents:
            self.candidate = None
            agent.action = None
            agent.penalty = agent.avg_penalty
            agent.best_penalty = agent.avg_penalty
            agent.long_term_penalty = 0
        self.not_assignment  = []

    def agent_observe(self, cid):
        i = self.cid2ind[cid]
        pos = self.agent_order_dict[cid]
        position = np.array([0 if i <= pos else 1 for i in range(len(self.agents))], dtype=np.float32)
        
        constraints = np.array([0 for _ in range(len(self.agents))], dtype=np.float32)
        if cid in self.not_assignment:
            constraints_cid = self.agents[i].hard_constraints_cids | self.agents[i].room_constraints_cids
        else:
            constraints_cid = self.agents[i].hard_constraints_cids | self.agents[i].soft_constraints_cids
        for ccid in list(constraints_cid):
            ind = self.agent_order_dict[ccid]
            ci = self.cid2ind[ccid]
            if self.agents[ci].action == None:
                constraints[ind] = 1
        obs = position * constraints
        return obs
    
    def agent_rewards(self, cid):
        # - [weight_a/N * len(no_assignment) + weight_p * penalty + weight_lt * long_term_penalty]
        i = self.cid2ind[cid]
        penalty = self.agents[i].penalty
        long_term_penalty = self.agents[i].long_term_penalty
        return self.weight_p * penalty + self.weight_lt * long_term_penalty
        # rewards = best_penalty - (penalty + long_term_penalty)
        # return rewards

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

    def is_feasible(self, cid, action):
        room_option_ind, time_option_ind, penalty = action
        self.Hard_validator.setClasses(self.agents)
        if room_option_ind != -1:
            i = self.cid2ind[cid]
            room_option = self.agents[i].room_options[room_option_ind]
            # same room & same time: two class
            if self.rooms[room_option['id']]['ocupied']:
                violate = self.Hard_validator.RoomConflicts(cid, self.rooms[room_option['id']]['ocupied'])
                if violate: return False
            # room unavailable
            violate = self.Hard_validator.RoomUnavailable(cid, self.rooms[room_option['id']]['unavailables_bits'])
            if violate: return False
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
        i = self.cid2ind[cid]
        self.not_assignment.append(cid)
        # observation update
        self.getRoomConflicts(cid)
        constraints_cid = self.agents[i].hard_constraints_cids | self.agents[i].room_constraints_cids
        # rewards update
        cis = []
        for ccid in constraints_cid:
            if self.agent_order_dict[ccid] < self.agent_order_dict[cid]:
                ci = self.cid2ind[ccid]
                cis.append(ci)
        self.agents[i].penalty = self.agents[i].avg_penalty
        for ci in cis:
            self.agents[ci].long_term_penalty += self.agents[i].avg_penalty

    def choose_action(self, cid, best_action, best_penalty, valid_actions):
        i = self.cid2ind[cid]
        if self.agents[i].choice == 0:
            action = best_action
            room_option_ind, time_option_ind, penalty = action
            self.agents[i].candidate = None
            self.agents[i].action = action
            self.agents[i].penalty = penalty
            time_option = self.agents[i].time_options[time_option_ind]
            if room_option_ind != -1:
                room_option = self.agents[i].room_options[room_option_ind]
                self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], penalty)) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}
        else:
            # ai = self.agents[i]._action_space.sample(mask=np.array(valid_actions, dtype=np.int8))
            # action = self.agents[i].action_space[ai]
            valid_action_space = [self.agents[i].action_space[j] for j, va in enumerate(valid_actions) if va == 1]
            action = valid_action_space[self.agents[i].choice % len(valid_action_space)]
            room_option_ind, time_option_ind, penalty = action
            self.agents[i].candidate = None
            self.agents[i].action = action
            self.agents[i].penalty = penalty
            time_option = self.agents[i].time_options[time_option_ind]
            if room_option_ind != -1:
                room_option = self.agents[i].room_options[room_option_ind]
                self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], penalty)) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}

    def total_penalty(self):
        penalty = 0
        Room_penalty = 0
        Time_penalty = 0
        observations = {}
        rewards = {}
        total_rewards = 0
        R1 = self.weight_a * len(self.not_assignment) // len(self.agents)
        for agent in self.agents:
            observations[agent.id] = self.agent_observe(agent.id)
            R = R1 + self.agent_rewards(agent.id)
            rewards[agent.id] = -R
            total_rewards -= R
            if agent.id not in self.not_assignment:
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
        return {
            "not assignment": self.not_assignment,
            "rewards": rewards,
            "total_rewards": total_rewards,
            "penalty": penalty,
            # "Total_cost": "TODO",
            # "Student conflicts": "TODO",
            "Time penalty": Time_penalty,
            "Room penalty": Room_penalty,
            "Distribution penalty": Distribution_penalty,
            "observations": observations
        }

    def step(self):
        # print("agents_order", list(self.agents_order.items())[:5])
        for i, (cid, value) in self.agents_order.items():
            i = self.cid2ind[cid]
            best_action = None
            valid_actions = []
            confilcts_cids = {}
            best_penalty = +inf

            for aid, action in enumerate(self.agents[i].action_space):
                self.agents[i].candidate = action
                feasible = self.is_feasible(cid, action)
                if not feasible:
                    valid_actions.append(0)
                    continue
                p = self.incremental_penalty(cid, action)

                if p < best_penalty:
                    best_penalty = p
                    best_action = action
                valid_actions.append(1)
            
            if best_action is None:
                self.handle_infeasible_case(cid)
                continue
            self.choose_action(cid, best_action, best_penalty, valid_actions)
        return self.total_penalty()

    def results(self):
        assignment = {}
        for agent in self.agents:
            result = agent.result()
            assignment.update(result)
        return assignment

    def get_agent_value(self, cid):
        i = self.cid2ind[cid]
        return self.agents[i].value

    def save(self, filename):
        print("model.save", self.total_penalty()["penalty"])
        data = {
            "classes":{agent.id: agent.action for agent in self.agents},
            "values":{agent.id: agent.value for agent in self.agents},
            "rooms": self.rooms
        }
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False)
