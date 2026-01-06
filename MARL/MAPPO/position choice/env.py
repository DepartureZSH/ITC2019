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

        # self.timeTable_matrix = self.reader.timeTable_matrix
        self.rooms = copy.deepcopy(self.reader.rooms) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}

        # self._action_space = spaces.Box(low=0, high=len(self.agents), shape=(len(self.agents),), dtype=np.float32)  # Placeholder
        # self.observation_spaces = spaces.Box(low=-1e9, high=1e9, shape=(len(self.agents),), dtype=np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self._assignment = []
        self.rooms = copy.deepcopy(self.reader.rooms)
        self.agents_order = self.order_agents() # (cid, value)
        observations = {}
        for agent in self.agents:
            agent.candidate = None
            agent.action = None
            agent.penalty = inf
            agent.choice = 0
            agent.masked_actions = [1 for _ in range(len(agent.action_space))]
            observations[agent.id] = self.agent_observe(agent.id)
            agent.observe_space = np.array([len(agent.action_space)/1 for _ in range(self.obs_shape)], dtype=np.float64)
        not_assignment = [agent.id for agent in self.agents]
        return observations, not_assignment
    
    def step_reset(self):
        self._assignment = []
        self.rooms = copy.deepcopy(self.reader.rooms)
        for agent in self.agents:
            agent.candidate = None
            agent.action = None
            agent.penalty = inf
            agent.observe_space = np.array([len(agent.action_space)/1 for _ in range(self.obs_shape)], dtype=np.float64)


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
    
    def move_agent(self, action_dict):
        for i, (cid, value) in enumerate(self.agents_order):
            if cid in self._assignment:
                self.agents_order[i] = (cid, action_dict[cid]/self.action_shape)
            else:
                self.agents[i].choice = action_dict[cid] % len(self.agents[i].action_space)
            # if action_dict[cid] == 3:
            #     i = self.cid2ind[cid]
            #     self.agents[i].choice = max(self.agents[i].choice -1, 0)
            # elif action_dict[cid] == 4:
            #     i = self.cid2ind[cid]
            #     self.agents[i].choice = min(self.agents[i].choice + 1, len(self.agents[i].action_space) - 1)
        agents_order = sorted(self.agents_order, key=lambda k:k[1])
        self.agent_dict = {cid: i for i, (cid, _) in enumerate(agents_order)}
        self.agents_order = {cid: (cid, i) if value!=0 else (cid, value) for i, (cid, value) in enumerate(agents_order)}

    def is_feasible(self, cid, action):
        room_option_ind, time_option_ind, penalty = action
        self.Hard_validator.setClasses(self.agents)
        confilct_agent_cid = []
        if room_option_ind != -1:
            i = self.cid2ind[cid]
            room_option = self.agents[i].room_options[room_option_ind]
            # TODO
            # 1. 同一间 room 在同一 time 不能被两个 class 占用
            if self.rooms[room_option['id']]['ocupied']:
                violate = self.Hard_validator.RoomConflicts(cid, self.rooms[room_option['id']]['ocupied'])
                self.check_agent(cid, ("RoomConflicts", violate))
                if violate: return False, confilct_agent_cid
            # ...
            # 2. room 自身的 unavailable 时间不能选
            violate = self.Hard_validator.RoomUnavailable(cid, self.rooms[room_option['id']]['unavailables_bits'])
            self.check_agent(cid, ("RoomUnavailable", violate), rid=room_option['id'])
            if violate: return False, confilct_agent_cid
        # 3. hard_constraints
        for hard_constraint in self.hard_constrains:
            if cid in hard_constraint['classes']:
                violate = self.Hard_validator._violation_rate(hard_constraint, cid)
                if violate: 
                    for ccid in hard_constraint['classes']:
                        if cid != ccid:
                            ci = self.cid2ind[ccid]
                            if self.agents[ci].action:
                                confilct_agent_cid.append(ccid)
                    self.check_agent(cid, (f"HardConstraint {hard_constraint}", violate))
                    return False, confilct_agent_cid
        return True, confilct_agent_cid

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
    
    def handle_infeasible_case(self, cid, confilcts_cids):
        self._assignment.append(cid)
        i = self.cid2ind[cid]
        
        action_len = len(self.agents[i].action_space)
        if confilcts_cids:
            competitor = max(confilcts_cids, key=lambda k: confilcts_cids[k])
            if self.agents[i].competitor == competitor:
                ci = self.cid2ind[competitor]
                self.agents[ci].choice = (self.agents[ci].choice + 1) % len(self.agents[ci].action_space)
            else:
                self.agents[i].competitor = competitor

    def apply_action(self, cid, best_action, best_penalty, valid_actions):
        i = self.cid2ind[cid]
        action = valid_actions[self.agents[i].choice % len(valid_actions)][0]
        room_option_ind, time_option_ind, penalty = action
        self.agents[i].candidate = None
        self.agents[i].action = action
        self.agents[i].penalty = penalty
        time_option = self.agents[i].time_options[time_option_ind]
        # self.timeTable_matrix = np.add(self.timeTable_matrix, time_option['optional_time'])
        if room_option_ind != -1:
            room_option = self.agents[i].room_options[room_option_ind]
            self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], penalty)) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}
         

    def total_penalty(self):
        penalty = 0
        Room_penalty = 0
        Time_penalty = 0
        observations = {}
        for agent in self.agents:
            if agent.id in self._assignment:
                observations[agent.id] =self.agent_observe(agent.id)
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
            "observations": observations
        }

    def step(self):
        # self.move_agent(action_dict)
        self.step_reset()
        # print(self.agents_order[:5])
        for cid, value in self.agents_order:
            i = self.cid2ind[cid]
            best_action = None
            valid_actions = []
            confilcts_cids = {}
            best_penalty = +inf

            c = 0
            for aid, action in enumerate(self.agents[i].action_space):
                self.agents[i].candidate = action
                feasible, confilcts = self.is_feasible(cid, action)
                if not feasible:
                    # self.agents[i].masked_actions[aid] = 0
                    for ccid in confilcts:
                        if confilcts_cids.get(ccid, 0): 
                            confilcts_cids[ccid] += 1
                        else: 
                            confilcts_cids[ccid] = 1
                    continue
                p = self.incremental_penalty(cid, action)

                if c < self.obs_shape:
                    self.agents[i].observe_space[c] = 1/p if p>0 else 1
                    c += 1
                # self.agents[i].masked_actions[aid] = 1

                if p < best_penalty:
                    best_penalty = p
                    best_action = action
                valid_actions.append((action, p))
            

            if best_action is None:
                self.agents[i].observe_space[:] = -1
                self.handle_infeasible_case(cid, confilcts_cids)
                continue
            else:
                valid_actions = sorted(valid_actions, key=lambda k:k[1])
                self.agents[i].observe_space[c:] = 1/best_penalty if best_penalty>0 else 1
                # break
            self.apply_action(cid, best_action, best_penalty, valid_actions)
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
            # values.append((cid, self.agents[i].value))
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