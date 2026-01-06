from math import inf
import numpy as np
from gymnasium.spaces import Discrete

class agent_class:
    def __init__(self, class_info, obs_shape=32):
        self.id = class_info['id']
        self.limit = class_info['limit']
        self.parent = class_info['parent']
        self.room_required = class_info['room_required']
        self.room_options = class_info['room_options']
        self.time_options = class_info['time_options']
        self.rooms = list(set([room_opt['id'] for room_opt in self.room_options]))

        self.action_space, self.total_penalty = self._actions()
        self.avg_penalty = self.total_penalty // len(self.action_space)
        self._action_space = Discrete(len(self.action_space), start=0, seed=42)
        self.observe_space = np.array([0 for _ in range(obs_shape)], dtype=np.float64) # conflict with other class
        
        self.value = len(self.action_space)
        self.choice = False
        self.candidate = None
        self.action = None
        self.penalty = self.avg_penalty
        self.best_penalty = self.avg_penalty
        self.long_term_penalty = 0

        self.room_constraints_cids = set()
        self.hard_constraints = 0
        self.hard_constraints_cids = set()
        self.soft_constraints = 0
        self.soft_constraints_cids = set()

    def _actions(self):
        actions = []
        total_penalty = 0
        if self.room_required:
            for i in range(len(self.room_options)):
                p1 = self.room_options[i]['penalty']
                for j in range(len(self.time_options)):
                    p2 = self.time_options[j]['penalty']
                    actions.append((i, j, p1 + p2))
                    total_penalty += p1 + p2
        else:
            for j in range(len(self.time_options)):
                p = self.time_options[j]['penalty']
                actions.append((-1, j, p))
                total_penalty += p
        actions = sorted(actions, key=lambda k:k[2])
        return actions, total_penalty
    
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