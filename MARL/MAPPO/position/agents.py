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
        self.action_space = self._actions()
        self._action_space = Discrete(len(self.action_space), start=0, seed=42)
        self.masked_actions = [1 for _ in range(len(self.action_space))]
        self.observe_space = np.array([len(self.action_space)/1 for _ in range(obs_shape)], dtype=np.float64) # penalty of each action
        self.value = len(self.action_space)
        self.candidate = None
        self.action = None
        self.include_students = False
        self.penalty = inf
        self.choice = 0
        self.ban = []
        self.competitor = "-1"

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