from math import inf
import numpy as np
from gymnasium.spaces import Discrete

class agent_class:
    def __init__(self, class_info, obs_shape=32, optimization=None):
        self.id = class_info['id']
        self.limit = class_info['limit']
        self.parent = class_info['parent']
        self.room_required = class_info['room_required']
        self.room_options = class_info['room_options']
        self.time_options = class_info['time_options']
        self.rooms = list(set([room_opt['id'] for room_opt in self.room_options]))
        
        self.optimization = optimization

        self.action_space, self.max_penalty, self.action_map = self._actions()
        self.action_penalty = np.array([0 for _ in range(len(self.action_space))], dtype=np.float64)
        self.masked_actions = np.array([1 for _ in range(len(self.action_space))], dtype=np.int8)
        self._action_space = Discrete(len(self.action_space), start=0, seed=42)
        self.observe_space = np.array([0 for _ in range(obs_shape)], dtype=np.float64) # conflict with other class
        
        self.value = len(self.action_space)
        self.choice = False
        self.candidate = None
        self.action = None
        self.last_action = None
        self.isfixed = False
        self.mask = np.array([0 for _ in range(len(self.action_space))], dtype=np.int8)
        self.penalty = self.max_penalty
        self.best_penalty = self.max_penalty
        self.long_term_penalty = 0

        self.room_constraints_cids = set()
        self.hard_constraints = 0
        self.hard_constraints_cids = set()
        self.soft_constraints = 0
        self.soft_constraints_cids = set()

    def _actions(self):
        action_map = {}
        actions = []
        max_penalty = 0
        if self.room_required:
            for i in range(len(self.room_options)):
                p1 = self.room_options[i]['penalty'] * self.optimization["room"]
                for j in range(len(self.time_options)):
                    p2 = self.time_options[j]['penalty'] * self.optimization["time"]
                    actions.append((i, j, p1 + p2))
                    max_penalty = max(max_penalty, p1 + p2)
        else:
            for j in range(len(self.time_options)):
                p = self.time_options[j]['penalty'] * self.optimization["time"]
                actions.append((-1, j, p))
                max_penalty = max(max_penalty, p)
        actions = sorted(actions, key=lambda k:k[2])
        for action in actions:
            weeks_bits, days_bits, start = self.time_options[action[1]]["optional_time_bits"][:3]
            if self.room_required:
                rid = self.room_options[action[0]]["id"]
                action_map[weeks_bits, days_bits, start, rid] = actions.index(action)
            else:
                action_map[weeks_bits, days_bits, start] = actions.index(action)
        return actions, max_penalty, action_map
    
    def get_action_idx(self, weeks_bits, days_bits, start, rid):
        if self.room_required:
            idx = self.action_map[weeks_bits, days_bits, start, rid]
        else:
            idx = self.action_map[weeks_bits, days_bits, start]
        return idx

    def result(self):
        if self.action == None:
            room_id = None
            topt = None
        elif self.room_required==False:
            room_id = None
            time_option_idx = self.action[1]
            topt = self.time_options[time_option_idx]
        else:
            oid = self.action[0]
            room_id = self.room_options[oid]['id']
            time_option_idx = self.action[1]
            topt = self.time_options[time_option_idx]
        return {self.id: (topt, self.room_required, room_id, None)}