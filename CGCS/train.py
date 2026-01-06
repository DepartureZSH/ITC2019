import pathlib
import yaml
import copy
import torch
import json
import numpy as np
from CGCS.constraints import HardConstraints, SoftConstraints
from math import inf
from tqdm import tqdm
folder = pathlib.Path(__file__).parent.resolve()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class agent:
    def __init__(self, class_info):
        self.id = class_info['id']
        self.limit = class_info['limit']
        self.parent = class_info['parent']
        self.room_required = class_info['room_required']
        self.room_options = class_info['room_options']
        self.time_options = class_info['time_options']
        self.action_space = self._actions()
        self.value = len(self.action_space)
        self.candidate = None
        self.action = None
        self.include_students = False
        self.penalty = inf

    def _actions(self):
        actions = []
        if self.room_required:
            for i in range(len(self.room_options)):
                p1 = self.room_options[i]['penalty']
                for j in range(len(self.time_options)):
                    p2 = self.time_options[j]['penalty']
                    actions.append((i, j, p1 + p2))
        else:
            for j in range(len(self.time_options)):
                p = self.time_options[j]['penalty']
                actions.append((-1, j, p))
        actions = sorted(actions, key=lambda k:k[2])
        return actions
    
    def result(self):
        if self.action == None:
            room_id = None
            topt = self.time_options[0]
        elif self.room_required==False:
            room_id = None
            time_option_idx = self.action[1]
            topt = self.time_options[time_option_idx]
            # return f"""<class id="{self.id}" days="{time_option[1]}" start="{time_option[2]}" weeks="{time_option[0]}"></class>"""
        else:
            oid = self.action[0]
            room_id = self.room_options[oid]['id']
            time_option_idx = self.action[1]
            topt = self.time_options[time_option_idx]
            # return f"""<class id="{self.id}" days="{time_option[1]}" start="{time_option[2]}" weeks="{time_option[0]}" room="{room_id}"></class>"""
        return {self.id: (topt, self.room_required, room_id, None)}
        
class trainer:
    def __init__(self, reader, discount=0.5):
        self.reader = reader
        self.discount = discount
        self.iter = 0
        
        self.hard_constrains = self.reader.distributions['hard_constraints']
        self.soft_constrains = self.reader.distributions['soft_constraints']
        self.Hard_validator = HardConstraints()
        self.Soft_validator = SoftConstraints()

        self.agents = [] # classes
        self.cid2ind = {}
        i = 0
        for key, each in self.reader.classes.items():
            self.agents.append(agent(each))
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
        self.not_assignment = []

        # self.timeTable_matrix = self.reader.timeTable_matrix
        self.rooms = copy.deepcopy(self.reader.rooms) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}

    def reset(self):
        self.not_assignment = []
        self.rooms = copy.deepcopy(self.reader.rooms)
        for agent in self.agents:
            agent.candidate = None
            agent.action = None
            agent.penalty = inf

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
            self.check_agent(cid, f"RoomUnavailable {violate}")
            if violate: return False
            pass
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
                # if soft_constrain['type']=="DifferentRoom": print(cid, f" type {type(cid)} ", soft_constrain)
                violation_rate = self.Soft_validator._violation_rate(soft_constrain, cid)
                if violation_rate:
                    p += violation_rate * soft_constrain['penalty']
        return p
    
    def handle_infeasible_case(self, cid):
        self.not_assignment.append(cid)
        i = self.cid2ind[cid]
        self.agents[i].value *= self.discount

    def apply_action(self, cid, best_action, best_penalty):
        room_option_ind, time_option_ind, penalty = best_action
        i = self.cid2ind[cid]
        self.agents[i].candidate = None
        self.agents[i].action = best_action
        self.agents[i].penalty = penalty
        time_option = self.agents[i].time_options[time_option_ind]
        # self.timeTable_matrix = np.add(self.timeTable_matrix, time_option['optional_time'])
        if room_option_ind != -1:
            room_option = self.agents[i].room_options[room_option_ind]
            self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], best_penalty)) # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}
        

    def total_penalty(self):
        penalty = 0
        Room_penalty = 0
        Time_penalty = 0
        for agent in self.agents:
            if agent.id not in self.not_assignment:
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
            "not assignment": self.not_assignment,
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
            best_action = None
            best_penalty = +inf

            for action in self.agents[i].action_space:
                self.agents[i].candidate = action
                if not self.is_feasible(cid, action):
                    continue

                p = self.incremental_penalty(cid, action)

                if p < best_penalty:
                    best_penalty = p
                    best_action = action
            
            if best_action is None:
                self.handle_infeasible_case(cid)
                continue
                # break
            self.apply_action(cid, best_action, best_penalty)
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
    
    def check_agent(self, cid, s):
        if cid == "-1":
            i = self.cid2ind[cid]
            print(f"agent {cid} {s}: ")
            print(f"  candidate: {self.agents[i].candidate}")
            print(f"  action: {self.agents[i].action}")
            print(f"  penalty: {self.agents[i].penalty}")
            print(f"  value: {self.agents[i].value}")
            print("")
    
    def save(self, filename):
        data = {
            "classes":{agent.id: agent.action for agent in self.agents},
            "rooms": self.rooms
        }
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False)
            


if __name__ == "__main__":
    # Load configuration
    config = load_cfg(f"{folder}/config.yaml")
    # Set device
    device = torch.device("cuda" if config['device'] == "gpu" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    file = config["data"]["file"]