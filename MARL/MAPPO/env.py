import pathlib
import yaml
import copy
import torch
import json
import numpy as np
from MARL.Random.agents import agent_class
from MARL.Random.constraints import HardConstraints, SoftConstraints
from math import inf
from tqdm import tqdm
from gymnasium import spaces
folder = pathlib.Path(__file__).parent.resolve()

class CustomEnvironment:
    def __init__(self, reader, discount=0.6):
        self.reader = reader
        self.iter = 0
        self.discount = discount
        
        self.hard_constrains = self.reader.distributions['hard_constraints']
        self.soft_constrains = self.reader.distributions['soft_constraints']
        self.Hard_validator = HardConstraints()
        self.Soft_validator = SoftConstraints()

        self.agents = [] # classes
        self.cid2ind = {}
        i = 0
        for key, each in self.reader.classes.items():
            self.agents.append(agent_class(each))
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
        self._assignment = []

        # self.timeTable_matrix = self.reader.timeTable_matrix
        self.rooms = copy.deepcopy(self.reader.rooms) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}

        # self._action_space = spaces.Box(low=0, high=len(self.agents), shape=(len(self.agents),), dtype=np.float32)  # Placeholder
        # self._obs_space = spaces.Box(low=-1e9, high=1e9, shape=(len(self.agents),), dtype=np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
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
    
    def agent_observe(self, cid):
        i = self.cid2ind[cid]
        obs = np.array(self.agents[i].observe_space, dtype=np.float32)
        return obs

    def order_agents(self):
        agents_value = [(agent.id, agent.value) for agent in self.agents]
        return sorted(agents_value, key=lambda k:k[1])

    def is_feasible(self, cid, action):
        room_option_ind, time_option_ind, penalty = action
        self.Hard_validator.setClasses(self.agents)
        if room_option_ind != -1:
            i = self.cid2ind[cid]
            room_option = self.agents[i].room_options[room_option_ind]
            # TODO
            # 1. 同一间 room 在同一 time 不能被两个 class 占用
            if self.rooms[room_option['id']]['ocupied']:
                violate = self.Hard_validator.RoomConflicts(cid, self.rooms[room_option['id']]['ocupied'])
                self.check_agent(cid, f"RoomConflicts {violate}")
                if violate: return False
            # ...
            # 2. room 自身的 unavailable 时间不能选
            violate = self.Hard_validator.RoomUnavailable(cid, self.rooms[room_option['id']]['unavailables_bits'])
            self.check_agent(cid, f"RoomUnavailable {violate}", rid=room_option['id'])
            if violate: return False
        # 3. hard_constraints
        for hard_constraint in self.hard_constrains:
            if cid in hard_constraint['classes']:
                violate = self.Hard_validator._violation_rate(hard_constraint, cid)
                if violate: 
                    self.check_agent(cid, f"HardConstraint {hard_constraint['type']} {violate}")
                    return False
        return True

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
    
    def handle_infeasible_case(self, cid):
        self._assignment.append(cid)
        i = self.cid2ind[cid]
        self.agents[i].value *= self.discount

    def apply_action(self, cid):
        i = self.cid2ind[cid]
        observe = self.agents[i].observe_space
        observe = np.max(observe) - observe + self.agents[i].masked_actions
        observe = np.array(observe, dtype=np.float64)
        observe = observe*self.agents[i].masked_actions
        observe = observe/np.sum(observe)
        action_ind = self.agents[i]._action_space.sample(probability=observe)
        action = self.agents[i].action_space[action_ind]
        if self.agents[i].masked_actions[action_ind] == 0:
            print(f"Error: infeasible action selected for agent {cid}")
            observe = self.agents[i].observe_space
            print("observe:", observe)
            observe = np.max(observe) - observe + self.agents[i].masked_actions
            print("observe after inversion:", observe)
            observe = np.array(observe, dtype=np.float64)
            observe = observe*self.agents[i].masked_actions
            print("observe after masking:", observe)
            observe = observe/np.sum(observe)
            print("final probabilities:", observe)
            print("masked actions:", self.agents[i].masked_actions)
            action = self.agents[i].action_space[action_ind]
            exit(1)
        self.agents[i].observe_space = observe.tolist()
        room_option_ind, time_option_ind, penalty = action
        self.agents[i].candidate = None
        self.agents[i].action = action
        self.agents[i].penalty = penalty
        time_option = self.agents[i].time_options[time_option_ind]
        # self.timeTable_matrix = np.add(self.timeTable_matrix, time_option['optional_time'])
        if room_option_ind != -1:
            room_option = self.agents[i].room_options[room_option_ind]
            self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], action)) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}
        

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
            "Distribution penalty": Distribution_penalty
        }

    def step(self, epoch_num=0):
        agents_order = self.order_agents() # (cid, value)
        for cid, value in tqdm(agents_order, desc=f"step {self.iter}", total=len(agents_order)):
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
            
            # if cid == "246":
            #     print("agent 246 masks:", self.agents[i].masked_actions)

            if not valid:
                self.handle_infeasible_case(cid)
                continue
                # break
            self.apply_action(cid)
        # self.check("Precedence")
        self.iter += 1
        return self.total_penalty()

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
        for cid in ids:
            i = self.cid2ind[cid]
            values.append((cid, self.agents[i].value))
        return values
    
    def check_agent(self, cid, s, rid=None):
        if cid == "-1":
            i = self.cid2ind[cid]
            print(f"agent {cid} {s}: ")
            if self.agents[i].room_options[self.agents[i].candidate[0]]['id'] == '3':
                print(f"  candidate: {self.agents[i].candidate}")
                print(f"  room options: {self.agents[i].room_options[self.agents[i].candidate[0]] if self.agents[i].candidate[0]!=-1 else 'N/A'}")
                print(f"  time options: {self.agents[i].time_options[self.agents[i].candidate[1]]['optional_time_bits']}")
                print(f"  action: {self.agents[i].action}")
                print(f"  penalty: {self.agents[i].penalty}")
                print(f"  value: {self.agents[i].value}")
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
        data = {
            "classes":{agent.id: agent.action for agent in self.agents},
            "rooms": self.rooms
        }
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False)