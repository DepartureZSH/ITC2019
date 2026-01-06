import pathlib
import yaml
import copy
import torch
import json
import numpy as np
from MARL.Random.constraints import HardConstraints, SoftConstraints
from dataReader import PSTTReader
from math import inf
from tqdm import tqdm
folder = pathlib.Path(__file__).parent.resolve()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class agent:
    def __init__(self, class_info, actions):
        self.id = class_info['id']
        self.limit = class_info['limit']
        self.parent = class_info['parent']
        self.room_required = class_info['room_required']
        self.room_options = class_info['room_options']
        self.time_options = class_info['time_options']
        self.action_space = self._actions()
        self.value = len(self.action_space)
        self.candidate = None
        self.action = actions[self.id]
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
        
class solution:
    def __init__(self, reader, dict_file):
        self.reader = reader
        with open(dict_file, "r") as f:
            solutions = json.load(f)
        actions = solutions['classes']
    
        self.hard_constrains = self.reader.distributions['hard_constraints']
        self.soft_constrains = self.reader.distributions['soft_constraints']
        self.Hard_validator = HardConstraints()
        self.Soft_validator = SoftConstraints()

        self.agents = [] # classes
        self.cid2ind = {}
        i = 0
        for key, each in self.reader.classes.items():
            self.agents.append(agent(each, actions))
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
        self.rooms = solutions['rooms'] # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}

    def total_penalty(self):
        penalty = 0
        Room_penalty = 0
        Time_penalty = 0
        Times = []
        Rooms = []
        for agent in self.agents:
            if agent.id not in self.not_assignment:
                penalty += agent.penalty
                if agent.action == None:
                    # print(f"agent {agent.id} not assigned!")
                    continue
                rid, tid, _ = agent.action
                if agent.room_required:
                    Room_penalty += agent.room_options[rid]['penalty']
                    if agent.room_options[rid]['penalty']:
                        Rooms.append((agent.id, f"Room {agent.room_options[rid]['id']}", agent.room_options[rid]['penalty']))
                Time_penalty += agent.time_options[tid]['penalty']
                if agent.time_options[tid]['penalty']:
                    Times.append((agent.id, agent.time_options[tid]['optional_time_bits'], agent.time_options[tid]['penalty']))
        self.Soft_validator.setClasses(self.agents)
        Distribution_penalty = 0
        Distributions = []
        for soft_constrain in self.soft_constrains:
            # TODO
            violation_rate = self.Soft_validator._violation_rate(soft_constrain)
            if violation_rate:
                Distributions.append((soft_constrain, violation_rate, soft_constrain['penalty'], violation_rate * soft_constrain['penalty']))
                Distribution_penalty += violation_rate * soft_constrain['penalty']
                penalty += violation_rate * soft_constrain['penalty']
        return {
            "not assignment": self.not_assignment,
            "penalty": penalty,
            "Total_cost": "TODO",
            "Student conflicts": "TODO",
            "Time penalty": Time_penalty,
            "Room penalty": Room_penalty,
            "Time details": Times,
            "Room details": Rooms,
            "Distribution penalty": Distribution_penalty,
            "Distributions": Distributions
        }

    def check_assignment(self, cid):
        i = self.cid2ind[cid]
        action = self.agents[i].action
        time_option = self.agents[i].time_options[action[1]]['optional_time_bits'] if action else None
        room_option = self.agents[i].room_options[action[0]] if action and self.agents[i].room_required else None
        print(f"check assignment for agent {cid}, time_option: {time_option} room_option: {room_option}")

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
            


if __name__ == "__main__":
    file = '/home/scxsz1/zsh/itc2019/data/late/muni-fi-fal17.xml'
    reader = PSTTReader(file)
    solu = solution(reader, '/home/scxsz1/zsh/Learning/MARL/PSTT/MARL/MAPPO/position/results/muni-fi-fal17.last.json')
    # solu.check_assignment("3")
    # solu.check_assignment("409")
    # solu.check_assignment("412")
    # solu.check_assignment("410")
    total = solu.total_penalty()
    print("Total", total["Distribution penalty"] + total["Time penalty"] + total["Room penalty"])
    print("Distribution penalty", total["Distribution penalty"])
    # for each in total["Distributions"]:
    #     print(each)
    print("Time penalty", total["Time penalty"])
    # for each in total["Time details"]:
    #     print(each)
    print("Room penalty", total["Room penalty"])
    # for each in total["Room details"]:
        # print(each)