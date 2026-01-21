import gurobipy as gp
from gurobipy import GRB
import itertools
from dataReader import PSTTReader
from Solution_writter import export_solution_xml

class GraphMIPSolver:
    def __init__(self, reader: PSTTReader, time_limit=300):
        self.reader = reader
        self.model = gp.Model("ITC2019_GraphMIP")
        self.model.setParam('TimeLimit', time_limit)
        
        # 1. Data Flattening & Preprocessing
        self.all_classes = [] # List of class_ids
        self.c_map = {}       # class_id -> class_dict
        self.r_map = {}       # room_id -> room_dict
        self._preprocess_data()
        
        # 2. Decision Variables Containers
        self.x = {} # x[class_id, time_idx, room_id] = Binary
        self.y = {} # y[class_id, time_idx] = Binary (Auxiliary for Time)
        self.w = {} # w[class_id, room_id] = Binary (Auxiliary for Room)
        self.soft_penalties = [] # List of penalty variables

    def _preprocess_data(self):
        """扁平化课程结构，方便建模"""
        for cid, c_data in self.reader.classes.items():
            self.all_classes.append(cid)
            self.c_map[cid] = c_data
        for rid, r_data in self.reader.rooms.items():
            self.r_map[rid] = r_data

    def _is_overlapping(self, t_opt1, t_opt2):
        """
        判断两个时间选项是否重叠
        t_opt: (weeks_bits, days_bits, start, length)
        """
        # 1. 检查周重叠 (按位与)
        w1, d1, s1, l1 = t_opt1
        w2, d2, s2, l2 = t_opt2
        
        weeks_overlap = (int(w1, 2) & int(w2, 2)) != 0
        days_overlap = (int(d1, 2) & int(d2, 2)) != 0
        
        # 2. 检查时间槽重叠
        e1 = s1 + l1
        e2 = s2 + l2
        slots_overlap = max(s1, s2) < min(e1, e2)
        
        return weeks_overlap and days_overlap and slots_overlap

    def build_model(self):
        print("Building Variables...")
        
        # --- 1. 定义变量 ---
        for cid in self.all_classes:
            c_data = self.c_map[cid]
            
            # y[c, t]: 课程 c 是否选择了时间选项 t
            for t_idx, t_opt in enumerate(c_data['time_options']):
                self.y[cid, t_idx] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{cid}_{t_idx}")
            
            # w[c, r]: 课程 c 是否选择了教室 r
            # 如果 room_required=False，我们通常还是分配一个虚拟房间或不分配(视具体逻辑)，
            # 这里简化为必须分配一个有效房间（通常题目中有Dummy room或允许空）
            # TODO
            if c_data['room_required']:
                for r_opt in c_data['room_options']:
                    rid = r_opt['id']
                    self.w[cid, rid] = self.model.addVar(vtype=GRB.BINARY, name=f"w_{cid}_{rid}")
            
            # x[c, t, r]: 核心连接变量
            if c_data['room_required']:
                for t_idx in range(len(c_data['time_options'])):
                    for r_opt in c_data['room_options']:
                        rid = r_opt['id']
                        self.x[cid, t_idx, rid] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{cid}_{t_idx}_{rid}")

        self.model.update()
        
        print("Building Basic Constraints...")
        # --- 2. 基础约束 ---
        for cid in self.all_classes:
            c_data = self.c_map[cid]
            
            # (1) 每门课必须选且仅选一个时间
            self.model.addConstr(gp.quicksum(self.y[cid, t] for t in range(len(c_data['time_options']))) == 1, 
                                 name=f"AssignTime_{cid}")
            
            if c_data['room_required']:
                # (2) 每门课必须选且仅选一个教室
                self.model.addConstr(gp.quicksum(self.w[cid, r['id']] for r in c_data['room_options']) == 1, 
                                     name=f"AssignRoom_{cid}")
                
                # (3) Linking Constraints: x, y, w 的关系
                # y[c,t] = sum_r x[c,t,r]
                for t_idx in range(len(c_data['time_options'])):
                    self.model.addConstr(
                        self.y[cid, t_idx] == gp.quicksum(self.x[cid, t_idx, r['id']] for r in c_data['room_options']),
                        name=f"Link_Y_{cid}_{t_idx}"
                    )
                
                # w[c,r] = sum_t x[c,t,r]
                for r_opt in c_data['room_options']:
                    rid = r_opt['id']
                    self.model.addConstr(
                        self.w[cid, rid] == gp.quicksum(self.x[cid, t, rid] for t in range(len(c_data['time_options']))),
                        name=f"Link_W_{cid}_{rid}"
                    )

        # --- 3. 房间冲突约束 (Room Conflict - Hard) ---
        print("Building Room Conflict Constraints...")
        room_usage = {} # room_id -> list of (class_id, valid_time_indices)
        
        for cid in self.all_classes:
            c_data = self.c_map[cid]
            if not c_data['room_required']: continue
            for r_opt in c_data['room_options']:
                rid = r_opt['id']
                if rid not in room_usage: room_usage[rid] = []
                room_usage[rid].append(cid)

        for rid, cids in room_usage.items():
            # 对在这个房间的每对课程
            # (1) 房间unavailable
            unavailables = self.r_map[rid]["unavailables_bits"]
            for cid in cids:
                opts = self.c_map[cid]['time_options']
                for t_idx, opt in enumerate(opts):
                    for unavailable_bit in unavailables:
                        if self._is_overlapping(opt['optional_time_bits'], unavailable_bit):
                                self.model.addConstr(
                                    self.x[cid, t_idx, rid] == 0,
                                    name=f"RoomUnavailable_{rid}_{cid}"
                                )

            # (2) 如果两门课使用同一个房间，且时间重叠，则禁止。
            for i in range(len(cids)):
                for j in range(i+1, len(cids)):
                    cid1 = cids[i]
                    cid2 = cids[j]
                    
                    # 检查这两门课的时间选项是否有重叠
                    opts1 = self.c_map[cid1]['time_options']
                    opts2 = self.c_map[cid2]['time_options']
                    
                    for t1_idx, opt1 in enumerate(opts1):
                        for t2_idx, opt2 in enumerate(opts2):
                            # 如果时间重叠，且都用了房间 rid -> 冲突
                            # x[c1,t1,rid] + x[c2,t2,rid] <= 1
                            if self._is_overlapping(opt1['optional_time_bits'], opt2['optional_time_bits']):
                                self.model.addConstr(
                                    self.x[cid1, t1_idx, rid] + self.x[cid2, t2_idx, rid] <= 1,
                                    name=f"RoomConf_{rid}_{cid1}_{cid2}"
                                )

        # --- 4. 分布约束 (Distribution Constraints) ---
        print("Building Distribution Constraints...")
        self._build_hard_constraints()
        self._build_soft_constraints()

        # --- 5. 目标函数 ---
        print("Building Objective...")
        obj = 0
        
        # (A) Time Penalties
        weight_time = self.reader.optimization['time']
        for cid in self.all_classes:
            opts = self.c_map[cid]['time_options']
            for t_idx, opt in enumerate(opts):
                if opt['penalty'] > 0:
                    obj += weight_time * opt['penalty'] * self.y[cid, t_idx]
        
        # (B) Room Penalties
        weight_room = self.reader.optimization['room']
        for cid in self.all_classes:
            c_data = self.c_map[cid]
            if c_data['room_required']:
                for r_opt in c_data['room_options']:
                    if r_opt['penalty'] > 0:
                        obj += weight_room * r_opt['penalty'] * self.w[cid, r_opt['id']]

        # (C) Distribution Penalties
        weight_dist = self.reader.optimization['distribution']
        for p_var, cost in self.soft_penalties:
            obj += weight_dist * cost * p_var

        self.model.setObjective(obj, GRB.MINIMIZE)

    def _build_hard_constraints(self):
        for constr in self.reader.distributions['hard_constraints']:
            ctype = constr['type']
            cids = constr['classes']
            
            # --- SameStart (Hard) ---
            if ctype == "SameStart":
                # 所有相关课程的 Start Time 必须相等
                # 简单实现：两两约束 (y_c1_t1 + y_c2_t2 <= 1 if start1 != start2)
                # 高级实现：按 StartTime 分组构建 Clique
                for i in range(len(cids)-1):
                    cid1, cid2 = cids[i], cids[i+1]
                    opts1 = self.c_map[cid1]['time_options']
                    opts2 = self.c_map[cid2]['time_options']
                    for t1, o1 in enumerate(opts1):
                        for t2, o2 in enumerate(opts2):
                            if o1['optional_time_bits'][2] != o2['optional_time_bits'][2]: # Start different
                                self.model.addConstr(self.y[cid1, t1] + self.y[cid2, t2] <= 1)

            # --- SameRoom (Hard) ---
            elif ctype == "SameRoom":
                # c1 和 c2 必须选同一个房间
                # w[c1, r] == w[c2, r] 
                # 或者： w[c1, r] + sum(w[c2, r_other]) <= 1
                for i in range(len(cids)-1):
                    cid1, cid2 = cids[i], cids[i+1]
                    common_rooms = set(r['id'] for r in self.c_map[cid1]['room_options']) & \
                                   set(r['id'] for r in self.c_map[cid2]['room_options'])
                    for r in common_rooms:
                        # 如果 c1 选了 r，c2 必须选 r (即 c2 不能选非 r)
                        # 这里简化为相等约束，前提是两者必须都选房间
                        if (cid1, r) in self.w and (cid2, r) in self.w:
                            self.model.addConstr(self.w[cid1, r] == self.w[cid2, r])

            # --- NotOverlap (Hard) ---
            elif ctype == "NotOverlap":
                for i in range(len(cids)):
                    for j in range(i+1, len(cids)):
                        cid1, cid2 = cids[i], cids[j]
                        opts1 = self.c_map[cid1]['time_options']
                        opts2 = self.c_map[cid2]['time_options']
                        for t1, o1 in enumerate(opts1):
                            for t2, o2 in enumerate(opts2):
                                if self._is_overlapping(o1['optional_time_bits'], o2['optional_time_bits']):
                                    self.model.addConstr(self.y[cid1, t1] + self.y[cid2, t2] <= 1)

            # --- Precedence (Hard) ---
            elif ctype == "Precedence":
                # c1 必须在 c2 之前 (First week, then day, then time)
                # 简化逻辑：只比较 end_time <= start_time (需要具体根据题目定义调整)
                for i in range(len(cids)-1):
                    cid1, cid2 = cids[i], cids[i+1]
                    opts1 = self.c_map[cid1]['time_options']
                    opts2 = self.c_map[cid2]['time_options']
                    for t1, o1 in enumerate(opts1):
                        for t2, o2 in enumerate(opts2):
                            # 这里简化判断：如果 o1 不在 o2 之前，则禁止
                            # 实际需要解析 bits 比较 week/day
                            # 假设只比较 start time (仅作演示)
                            s1 = o1['optional_time_bits'][2]
                            l1 = o1['optional_time_bits'][3]
                            s2 = o2['optional_time_bits'][2]
                            # 如果 c1 结束时间 > c2 开始时间，则禁止 (y1+y2<=1)
                            if s1 + l1 > s2: 
                                self.model.addConstr(self.y[cid1, t1] + self.y[cid2, t2] <= 1)

    def _build_soft_constraints(self):
        for constr in self.reader.distributions['soft_constraints']:
            penalty_cost = constr['penalty']
            ctype = constr['type']
            cids = constr['classes']
            
            # 这里的实现策略：
            # 引入一个二元变量 p 表示是否违反
            # 约束： 违反逻辑 <= p
            # 目标函数： + cost * p
            
            if ctype == "SameStart":
                # 两两检查，如果不相等，则罚分
                for i in range(len(cids)):
                    for j in range(i+1, len(cids)):
                        p_var = self.model.addVar(vtype=GRB.BINARY)
                        self.soft_penalties.append((p_var, penalty_cost))
                        
                        cid1, cid2 = cids[i], cids[j]
                        opts1 = self.c_map[cid1]['time_options']
                        opts2 = self.c_map[cid2]['time_options']
                        
                        # 逻辑：如果 y1=1 且 y2=1 且 start1 != start2，则 p=1
                        # 线性化： y1 + y2 - 1 <= p (当且仅当 start不同时添加此约束)
                        for t1, o1 in enumerate(opts1):
                            for t2, o2 in enumerate(opts2):
                                if o1['optional_time_bits'][2] != o2['optional_time_bits'][2]:
                                    self.model.addConstr(self.y[cid1, t1] + self.y[cid2, t2] - 1 <= p_var)

            # 其他软约束逻辑类似...

    def solve(self, out_xml_path="solution.xml"):
        self.build_model()
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL or self.model.Status == GRB.TIME_LIMIT:
            print(f"Solution Found! Obj: {self.model.ObjVal}")
            assignments = {}
            
            for cid in self.all_classes:
                c_data = self.c_map[cid]
                # Find chosen time
                chosen_t_idx = -1
                for t_idx in range(len(c_data['time_options'])):
                    if self.y[cid, t_idx].X > 0.5:
                        chosen_t_idx = t_idx
                        break
                
                # Find chosen room
                chosen_r_id = None
                if c_data['room_required']:
                    for r_opt in c_data['room_options']:
                        rid = r_opt['id']
                        if self.w[cid, rid].X > 0.5:
                            chosen_r_id = rid
                            break
                
                time_info = c_data['time_options'][chosen_t_idx] if chosen_t_idx != -1 else None
                assignments[cid] = (time_info, c_data['room_required'], chosen_r_id, [])
            
            # Export
            export_solution_xml(
                assignments, out_xml_path, 
                name=self.reader.problem_name,
                runtime_sec=self.model.Runtime,
                cores=8,
                technique="Gurobi MIP",
                author="AI",
                institution="Gemini",
                country="Cloud"
            )
        else:
            print("No solution found.")

if __name__ == "__main__":
    # 使用提供的 lums-sum17.xml 路径
    file_path = "/home/scxsz1/zsh/Learning/MARL/PSTT/data/early/muni-fi-spr16.xml" # 确保此文件在当前目录下或修改路径
    
    try:
        reader = PSTTReader(file_path)
        
        solver = GraphMIPSolver(reader, time_limit=600)
        solver.solve("muni-fi-spr16-solution.xml")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()