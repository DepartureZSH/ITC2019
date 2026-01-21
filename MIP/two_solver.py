import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict
from itertools import combinations
import time
import torch
from tqdm import tqdm
from dataReader import PSTTReader
from Solution_writter import export_solution_xml

class TwoStepSolverTimeFirst:
    """
    Two-step solver: Step 1 assigns times, Step 2 assigns rooms
    """
    def __init__(self, reader, logger, config=None):
        self.reader = reader
        self.logger = logger
        self.max_backtracks = config['train']['MIP']['max_backtracks']
        self.time_limit = config['train']['MIP']['time_limit']
        self.Threads = config['train']['MIP']['Threads']
        self.MIPGap = config['train']['MIP']['MIPGap']
        self.PoolSolutions = config['train']['MIP']['PoolSolutions']
        
        # Step 1: Time assignment model
        self.model_time = gp.Model("TimeAssignment")
        self.model_time.setParam('TimeLimit', self.time_limit)
        self.model_time.setParam('Threads', self.Threads)
        self.model_time.setParam('MIPGap', self.MIPGap)
        self.model_time.setParam('MIPFocus', 0)
        self.model_time.setParam('PoolSearchMode', 2)
        self.model_time.setParam('PoolSolutions', self.PoolSolutions)
        
        # Decision variables
        self.y = {}  # y[cid, time_idx]: class-time assignment
        # self.u_time = {}  # unassigned in time step
        
        # Helper variables
        self.penalty_vars_time = []
        
        # Index mapping
        self.class_to_time_options = {}
        self.class_to_room_options = {}
        self.class_to_value = {}
        self.time_conflict_cache = {}
        
        # Step 1 results
        self.time_assignments_results = []  # [{cid: (topt, tidx)}]
        self.time_assignments = {}  # {cid: (topt, tidx)}
        # Step 2 results
        self.room_assignments = {} # {cid: rid}
        
        self.logger.info(f"Initialized two-step solver (time-first) for: {self.reader.problem_name}")
        self.logger.info(f"Classes: {len(self.reader.classes)}, Rooms: {len(self.reader.rooms)}")

    def build_and_solve(self):
        """Build and solve both steps"""
        print("\n=== Two-Step Solver: Time First ===")
        
        # Build indices
        self._build_indices()
        
        # Step 1: Solve time assignment
        print("\n--- Step 1: Time Assignment ---")
        self._build_time_model()
        res1 = self._solve_time_step()
        
        if not res1:
            self.logger.error("Step 1 failed: no time assignments found")
            return None
        
        # Step 2: Solve room assignment given time assignments
        print("\n--- Step 2: Room Assignment ---")
        valid = False
        for it in range(self.model_time.SolCount):
            self.next_solution(it)
            self._build_room_model()
            res2 = self._solve_room_step()

            if res2:
                assigned_count = len(self.room_assignments)
                total_count = len(self.reader.classes)
                if assigned_count == total_count:
                    valid = True
                    break
        if valid:
            return self.extract_solution()
        else:
            self.add_room_conflict_cut()
            self.model_time.reset()
            res = self.solve_with_backtracking()
            if res: return self.extract_solution()
            return None
    
    def solve_with_backtracking(self):
        for it in range(self.max_backtracks):
            self.logger.info(f"\n=== Iteration {it} ===")

            # Step 1: Time
            self.logger.info(f"Time model: {self.model_time.NumVars} vars, {self.model_time.NumConstrs} constrs")
            self.time_assignments_results = []
            self.time_assignments = {}
            res1 = self._solve_time_step()
            if not res1:
                self.logger.error("Step 1 failed: no time assignments found")
                return False

            # Step 2: Room
            valid = False
            for it in range(self.model_time.SolCount):
                self.next_solution(it)
                self._build_room_model()
                res2 = self._solve_room_step()

                if res2:
                    assigned_count = len(self.room_assignments)
                    total_count = len(self.reader.classes)
                    if assigned_count < total_count:
                        self.add_room_conflict_cut()
                        self.model_time.reset()
                    else:
                        return True
                else:
                    self.add_room_conflict_cut()
                    self.model_time.reset()
        self.logger.error("Backtracking limit reached")
        return False
    
    def _build_indices(self, value=None):
        """Build class to time/room options mapping"""
        print("Building indices...")
        opt_count = {}
        self.max_count = 0
        for cid, class_data in self.reader.classes.items():
            time_options = []
            for idx, topt in enumerate(class_data['time_options']):
                time_options.append((topt, idx))
            self.class_to_time_options[cid] = time_options
            
            room_options = []
            for ropt in class_data['room_options']:
                room_options.append(ropt['id'])
            
            if not class_data['room_required']:
                room_options = ['dummy']
            count = len(time_options) * len(room_options)
            opt_count[cid] = count
            self.max_count = max(self.max_count, count)
            self.class_to_room_options[cid] = room_options

        for cid, class_data in self.reader.classes.items():
            if value:
                self.class_to_value[cid] = value[cid]
            else:
                self.class_to_value[cid] = self.max_count - opt_count[cid]
        
        print(f"Indexed {len(self.class_to_time_options)} classes")
    
    def _build_time_model(self):
        """Build MIP model for time assignment only"""
        print("Building time assignment model...")
        
        # Create variables
        for cid in self.reader.classes.keys():
            time_options = self.class_to_time_options[cid]
            
            # self.u_time[cid] = self.model_time.addVar(
            #     vtype=GRB.BINARY,
            #     name=f"u_time_{cid}"
            # )
            
            for topt, tidx in time_options:
                self.y[cid, tidx] = self.model_time.addVar(
                    vtype=GRB.BINARY,
                    name=f"y_{cid}_{tidx}"
                )
                self.y[cid, tidx].setAttr("BranchPriority", self.class_to_value[cid])
        
        # Constraints: each class assigned to exactly one time or unassigned
        for cid in self.reader.classes.keys():
            time_options = self.class_to_time_options[cid]
            y_vars = [self.y[cid, tidx] for _, tidx in time_options]
            
            self.model_time.addConstr(
                # gp.quicksum(y_vars) + self.u_time[cid] == 1,
                gp.quicksum(y_vars) == 1,
                name=f"assign_time_{cid}"
            )
        
        self.time_options_filter()

        # Room-independent constraints (time conflicts, attendee constraints, etc.)
        self._add_time_constraints()
        
        # Objective: minimize unassigned + penalties
        self._set_time_objective()
        
        print(f"Time model: {self.model_time.NumVars} vars, {self.model_time.NumConstrs} constrs")
    
    def time_options_filter(self):
        for cid in self.reader.classes.keys():
            time_options = self.class_to_time_options[cid]
            room_opts = self.class_to_room_options[cid]
            for topt, tidx in time_options:
                bits1 = topt['optional_time_bits']
                conflicts = True
                for rid in room_opts:
                    if rid == "dummy" or len(self.reader.rooms[rid]['unavailables_bits']) == 0:
                        conflicts = False
                        break
                    for bits2 in self.reader.rooms[rid]['unavailables_bits']:
                        if not self._times_conflict(bits1, bits2):
                            conflicts = False
                            break
                    if not conflicts:
                        break
                if conflicts:
                    self.model_time.addConstr(
                        self.y[cid, tidx] <= 0,
                        name=f"time_all_room_unavailable_{cid}_{tidx}"
                    )


    def _add_time_constraints(self):
        """Add time-related distribution constraints"""
        print("Adding time constraints...")
        
        for constraint in self.reader.distributions['hard_constraints']:
            self._add_time_constraint(constraint, is_hard=True)
        
        # for constraint in self.reader.distributions['soft_constraints']:
        #     self._add_time_constraint(constraint, is_hard=False)
    
    def _add_time_constraint(self, constraint, is_hard):
        """Add a single time-related constraint"""
        ctype = constraint['type']
        classes = constraint['classes']
        penalty = constraint.get('penalty', 0)
        
        if ctype == 'SameTime':
            self._add_same_time_constraint(classes, is_hard, penalty)
        elif ctype == 'DifferentTime':
            self._add_different_time_constraint(classes, is_hard, penalty)
        elif ctype == 'NotOverlap':
            self._add_not_overlap_constraint(classes, is_hard, penalty)
        elif ctype == 'Overlap':
            self._add_overlap_constraint(classes, is_hard, penalty)
        elif ctype == 'SameStart':
            self._add_same_start_constraint(classes, is_hard, penalty)
        elif ctype == 'SameDays':
            self._add_same_days_constraint(classes, is_hard, penalty)
        elif ctype == 'DifferentDays':
            self._add_different_days_constraint(classes, is_hard, penalty)
        elif ctype == 'SameWeeks':
            self._add_same_weeks_constraint(classes, is_hard, penalty)
        elif ctype == 'DifferentWeeks':
            self._add_different_weeks_constraint(classes, is_hard, penalty)
        elif ctype == 'Precedence':
            self._add_precedence_constraint(classes, is_hard, penalty)
        elif ctype.startswith('MinGap'):
            min_gap = int(ctype.split('(')[1].rstrip(')'))
            self._add_min_gap_constraint(classes, min_gap, is_hard, penalty)
        elif ctype.startswith('MaxDays'):
            max_days = int(ctype.split('(')[1].rstrip(')'))
            self._add_max_days_constraint(classes, max_days, is_hard, penalty)
        elif ctype.startswith('MaxDayLoad'):
            max_slots = int(ctype.split('(')[1].rstrip(')'))
            self._add_max_day_load_constraint(classes, max_slots, is_hard, penalty)
        elif ctype.startswith('WorkDay'):
            max_slots = int(ctype.split('(')[1].rstrip(')'))
            self._add_workday_constraint(classes, max_slots, is_hard, penalty)
        elif ctype.startswith('MaxBreaks'):
            params = ctype.split('(')[1].rstrip(')').split(',')
            max_breaks = int(params[0])
            min_break_length = int(params[1])
            self._add_max_breaks_constraint(classes, max_breaks, min_break_length, is_hard, penalty)
        elif ctype.startswith('MaxBlock'):
            params = ctype.split('(')[1].rstrip(')').split(',')
            max_block_length = int(params[0])
            max_gap_in_block = int(params[1])
            self._add_max_block_constraint(classes, max_block_length, max_gap_in_block, is_hard, penalty)
        # elif ctype == 'SameAttendees': # room related
        #     # SameAttendees only checks time overlap in this step
        #     self._add_same_attendees_constraint(classes, is_hard, penalty)
        #     self._add_not_overlap_constraint(classes, is_hard, penalty)
    
    def _times_conflict(self, time_bits1, time_bits2):
        """Check if two times conflict using bit operations"""
        if time_bits1 < time_bits2:
            cache_key = (time_bits1, time_bits2)
        else:
            cache_key = (time_bits2, time_bits1)
        
        if cache_key in self.time_conflict_cache:
            return self.time_conflict_cache[cache_key]
        
        week_bits1, day_bits1, start1, length1 = time_bits1
        week_bits2, day_bits2, start2, length2 = time_bits2
        
        end1 = start1 + length1
        end2 = start2 + length2
        
        if not ((start1 < end2) and (start2 < end1)):
            self.time_conflict_cache[cache_key] = False
            return False
        
        days_int1 = int(day_bits1, 2)
        days_int2 = int(day_bits2, 2)
        if (days_int1 & days_int2) == 0:
            self.time_conflict_cache[cache_key] = False
            return False
        
        week_int1 = int(week_bits1, 2)
        week_int2 = int(week_bits2, 2)
        if (week_int1 & week_int2) == 0:
            self.time_conflict_cache[cache_key] = False
            return False
        
        self.time_conflict_cache[cache_key] = True
        return True
    
    def _add_same_time_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    _, _, start1, end1 = topt1["optional_time_bits"]
                    _, _, start2, end2 = topt2["optional_time_bits"]
                    if start1 <= start2 and start2 + end2 <= start1 + end1:
                        continue
                    elif start2 <= start1 and start1 + end1 <= start2 + end2:
                        continue
                    else:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1,
                                name=f"same_time_{c1}_{c2}_{tidx1}_{tidx2}"
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_different_time_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    _, _, start1, end1 = topt1['optional_time_bits']
                    _, _, start2, end2 = topt2['optional_time_bits']
                    if (start1 + end1 <= start2) or (start2 + end2 <= start1):
                        continue
                    else:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_not_overlap_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
                    week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']
                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2
                    if (start1 + end1 <= start2) or (start2 + end2 <= start1) or (and_days == 0) or (and_week == 0):
                        continue
                    else:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_overlap_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
                    week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']
                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2
                    if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0):
                        continue
                    else:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    # def _add_same_attendees_constraint(self, classes, is_hard, penalty):
    #     if len(classes) < 2:
    #         return
        
    #     for c1, c2 in combinations(classes, 2):
    #         if c1 not in self.reader.classes or c2 not in self.reader.classes:
    #             continue
            
    #         time_opts1 = self.class_to_time_options[c1]
    #         time_opts2 = self.class_to_time_options[c2]
            
    #         for topt1, tidx1 in time_opts1:
    #             for topt2, tidx2 in time_opts2:
    #                 week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
    #                 week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']
    #                 days_int1 = int(day_bits1, 2)
    #                 days_int2 = int(day_bits2, 2)
    #                 and_days = days_int1 & days_int2
    #                 week_int1 = int(week_bits1, 2)
    #                 week_int2 = int(week_bits2, 2)
    #                 and_week = week_int1 & week_int2
    #                 if (start1 + end1 <= start2) or (start2 + end2 <= start1) or (and_days == 0) or (and_week == 0):
    #                     continue
    #                 else:
    #                     if is_hard:
    #                         self.model_time.addConstr(
    #                             self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
    #                         )
    #                     else:
    #                         p = self.model_time.addVar(vtype=GRB.BINARY)
    #                         self.model_time.addConstr(
    #                             self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
    #                         )
    #                         self.penalty_vars_time.append((p, penalty))

    def _add_same_start_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    if topt1['optional_time_bits'][2] != topt2['optional_time_bits'][2]:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_same_days_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    day_bits1 = topt1['optional_time_bits'][1]
                    days_int1 = int(day_bits1, 2)
                    day_bits2 = topt2['optional_time_bits'][1]
                    days_int2 = int(day_bits2, 2)
                    or_ = days_int1 | days_int2
                    if not (or_ == days_int1 or or_ == day_bits2):
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_different_days_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    day_bits1 = topt1['optional_time_bits'][1]
                    days_int1 = int(day_bits1, 2)
                    day_bits2 = topt2['optional_time_bits'][1]
                    days_int2 = int(day_bits2, 2)
                    and_ = days_int1 & days_int2
                    if not and_ == 0:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_same_weeks_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1 = topt1['optional_time_bits'][0]
                    week_int1 = int(week_bits1, 2)
                    week_bits2 = topt2['optional_time_bits'][0]
                    week_int2 = int(week_bits2, 2)
                    or_ = week_int1 | week_int2
                    if not (or_ == week_int1 or or_ == week_int2):
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_different_weeks_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1 = topt1['optional_time_bits'][0]
                    week_int1 = int(week_bits1, 2)
                    week_bits2 = topt2['optional_time_bits'][0]
                    week_int2 = int(week_bits2, 2)
                    and_ = week_int1 & week_int2
                    if not and_ == 0:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_precedence_constraint(self, classes, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for i in range(len(classes) - 1):
            c1 = classes[i]
            c2 = classes[i + 1]
            
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
                    week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']
                    first_day1 = day_bits1.find('1')
                    first_day2 = day_bits2.find('1')
                    first_week1 = week_bits1.find('1')
                    first_week2 = week_bits2.find('1')
                    w_pre, d_pre, s_pre, e_pre = first_week1, first_day1, start1, end1
                    w_sub, d_sub, s_sub, e_sub = first_week2, first_day2, start2, end2
                    if (w_pre < w_sub) or ( # first(week_i) < first(week_j) or
                        (w_pre == w_sub) and (
                            (d_pre < d_sub ) or ( # first(day_i) < first(day_j) or
                                (d_pre == d_sub) and (s_pre+e_pre <= s_sub) # end_i <= start_j
                            )
                        )
                    ):
                        continue
                    else:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_min_gap_constraint(self, classes, min_gap, is_hard, penalty):
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                bits1 = topt1['optional_time_bits']
                week_bits1, day_bits1, start1, end1 = bits1
                
                for topt2, tidx2 in time_opts2:
                    bits2 = topt2['optional_time_bits']
                    week_bits2, day_bits2, start2, end2 = bits2
                    
                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2
                    
                    if (and_days == 0) or (and_week == 0) or (start1 + end1 + min_gap <= start2) or (start2 + end2 + min_gap <= start1):
                        continue
                    else:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_max_days_constraint(self, classes, max_days, is_hard, penalty):
        if len(classes) == 0:
            return
        
        gamma = {}
        for d in range(self.reader.nrDays):
            gamma[d] = self.model_time.addVar(vtype=GRB.BINARY, name=f"gamma_day_{d}")
        
        M = len(classes)
        for d in range(self.reader.nrDays):
            day_vars = []
            for cid in classes:
                if cid not in self.reader.classes:
                    continue
                time_opts = self.class_to_time_options[cid]
                for topt, tidx in time_opts:
                    days_bits = topt['optional_time_bits'][1]
                    if d < len(days_bits) and days_bits[d] == '1':
                        day_vars.append(self.y[cid, tidx])
            
            if day_vars:
                self.model_time.addConstr(
                    gp.quicksum(day_vars) <= M * gamma[d]
                )
        
        if is_hard:
            self.model_time.addConstr(
                gp.quicksum(gamma[d] for d in range(self.reader.nrDays)) <= max_days
            )
        else:
            p = self.model_time.addVar(vtype=GRB.INTEGER, lb=0)
            self.model_time.addConstr(
                gp.quicksum(gamma[d] for d in range(self.reader.nrDays)) - max_days <= p
            )
            self.penalty_vars_time.append((p, penalty))
    
    def _add_max_day_load_constraint(self, classes, max_slots, is_hard, penalty):
        if len(classes) == 0:
            return
        
        for w in range(self.reader.nrWeeks):
            for d in range(self.reader.nrDays):
                day_load = []
                for cid in classes:
                    if cid not in self.reader.classes:
                        continue
                    time_opts = self.class_to_time_options[cid]
                    for topt, tidx in time_opts:
                        bits = topt['optional_time_bits']
                        weeks_bits, days_bits, start, length = bits
                        if (w < len(weeks_bits) and weeks_bits[w] == '1' and
                            d < len(days_bits) and days_bits[d] == '1'):
                            day_load.append(length * self.y[cid, tidx])
                
                if day_load:
                    if is_hard:
                        self.model_time.addConstr(
                            gp.quicksum(day_load) <= max_slots
                        )
                    else:
                        p = self.model_time.addVar(vtype=GRB.INTEGER, lb=0)
                        self.model_time.addConstr(
                            gp.quicksum(day_load) - max_slots <= p
                        )
                        self.penalty_vars_time.append((p, penalty))
    
    def _add_workday_constraint(self, classes, max_slots, is_hard, penalty):
        if len(classes) == 0:
            return
        
        for w in range(self.reader.nrWeeks):
            for d in range(self.reader.nrDays):
                day_events = []
                
                for cid in classes:
                    if cid not in self.reader.classes:
                        continue
                    
                    time_opts = self.class_to_time_options[cid]
                    for topt, tidx in time_opts:
                        bits = topt['optional_time_bits']
                        weeks_bits, days_bits, start, length = bits
                        
                        if (w < len(weeks_bits) and weeks_bits[w] == '1' and
                            d < len(days_bits) and days_bits[d] == '1'):
                            end = start + length
                            day_events.append((cid, tidx, start, end))
                
                if len(day_events) == 0:
                    continue
                
                if len(day_events) == 1:
                    cid, tidx, start, end = day_events[0]
                    workday_length = end - start
                    
                    if workday_length > max_slots:
                        if is_hard:
                            self.model_time.addConstr(
                                self.y[cid, tidx] == 0
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(self.y[cid, tidx] <= p)
                            self.penalty_vars_time.append((p, penalty * (workday_length - max_slots)))
                    continue
                
                for i in range(len(day_events)):
                    for j in range(i + 1, len(day_events)):
                        c1, t1, start1, end1 = day_events[i]
                        c2, t2, start2, end2 = day_events[j]
                        
                        earliest_start = min(start1, start2)
                        latest_end = max(end1, end2)
                        workday_length = latest_end - earliest_start
                        
                        if workday_length > max_slots:
                            if is_hard:
                                self.model_time.addConstr(
                                    self.y[c1, t1] + self.y[c2, t2] <= 1
                                )
                            else:
                                p = self.model_time.addVar(vtype=GRB.BINARY)
                                self.model_time.addConstr(
                                    self.y[c1, t1] + self.y[c2, t2] - 1 <= p
                                )
                                self.penalty_vars_time.append((p, penalty))
                
                if len(day_events) >= 3:
                    all_starts = [start for _, _, start, _ in day_events]
                    all_ends = [end for _, _, _, end in day_events]
                    total_span = max(all_ends) - min(all_starts)
                    
                    if total_span > max_slots:
                        if is_hard:
                            self.model_time.addConstr(
                                gp.quicksum(self.y[c, t] for c, t, _, _ in day_events) 
                                <= len(day_events) - 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                gp.quicksum(self.y[c, t] for c, t, _, _ in day_events)
                                - len(day_events) + 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _add_max_breaks_constraint(self, classes, max_breaks, min_break_length, is_hard, penalty):
        if len(classes) == 0:
            return
        
        for w in range(self.reader.nrWeeks):
            for d in range(self.reader.nrDays):
                day_events = []
                
                for cid in classes:
                    if cid not in self.reader.classes:
                        continue
                    
                    time_opts = self.class_to_time_options[cid]
                    for topt, tidx in time_opts:
                        bits = topt['optional_time_bits']
                        weeks_bits, days_bits, start, length = bits
                        
                        if (w < len(weeks_bits) and weeks_bits[w] == '1' and
                            d < len(days_bits) and days_bits[d] == '1'):
                            day_events.append((cid, tidx, start, start + length))
                
                if len(day_events) < 2:
                    continue
                
                day_events.sort(key=lambda x: x[2])
                
                break_vars = []
                
                for i in range(len(day_events) - 1):
                    c1, t1, start1, end1 = day_events[i]
                    c2, t2, start2, end2 = day_events[i + 1]
                    
                    gap = start2 - end1
                    
                    if gap > min_break_length:
                        b = self.model_time.addVar(vtype=GRB.BINARY)
                        
                        self.model_time.addConstr(
                            self.y[c1, t1] + self.y[c2, t2] - 1 <= b
                        )
                        self.model_time.addConstr(
                            b <= self.y[c1, t1]
                        )
                        self.model_time.addConstr(
                            b <= self.y[c2, t2]
                        )
                        
                        break_vars.append(b)
                
                if break_vars:
                    if is_hard:
                        self.model_time.addConstr(
                            gp.quicksum(break_vars) <= max_breaks
                        )
                    else:
                        p = self.model_time.addVar(vtype=GRB.INTEGER, lb=0)
                        self.model_time.addConstr(
                            gp.quicksum(break_vars) - max_breaks <= p
                        )
                        self.penalty_vars_time.append((p, penalty))
    
    def _add_max_block_constraint(self, classes, max_block_length, max_gap_in_block, is_hard, penalty):
        if len(classes) == 0:
            return
        
        for w in range(self.reader.nrWeeks):
            for d in range(self.reader.nrDays):
                day_events = []
                
                for cid in classes:
                    if cid not in self.reader.classes:
                        continue
                    
                    time_opts = self.class_to_time_options[cid]
                    for topt, tidx in time_opts:
                        bits = topt['optional_time_bits']
                        weeks_bits, days_bits, start, length = bits
                        
                        if (w < len(weeks_bits) and weeks_bits[w] == '1' and
                            d < len(days_bits) and days_bits[d] == '1'):
                            day_events.append((cid, tidx, start, start + length, length))
                
                if len(day_events) < 2:
                    continue
                
                day_events.sort(key=lambda x: x[2])
                
                for i in range(len(day_events)):
                    block_events = [day_events[i]]
                    block_start = day_events[i][2]
                    block_end = day_events[i][3]
                    
                    for j in range(i + 1, len(day_events)):
                        c_next, t_next, start_next, end_next, len_next = day_events[j]
                        
                        gap = start_next - block_end
                        
                        if gap < max_gap_in_block:
                            block_events.append(day_events[j])
                            block_end = max(block_end, end_next)
                        else:
                            break
                    
                    block_length = block_end - block_start
                    
                    if block_length > max_block_length and len(block_events) > 1:
                        if is_hard:
                            self.model_time.addConstr(
                                gp.quicksum(self.y[c, t] for c, t, _, _, _ in block_events) 
                                <= len(block_events) - 1
                            )
                        else:
                            p = self.model_time.addVar(vtype=GRB.BINARY)
                            self.model_time.addConstr(
                                gp.quicksum(self.y[c, t] for c, t, _, _, _ in block_events)
                                - len(block_events) + 1 <= p
                            )
                            self.penalty_vars_time.append((p, penalty))
    
    def _set_time_objective(self):
        """Set objective for time assignment step"""
        print("Setting time objective...")
        
        BIG_M = 100000

        # Priority 1: Minimize unassigned classes
        # unassigned_obj = gp.quicksum(self.u_time[cid] for cid in self.reader.classes.keys())
        # self.model_time.setObjectiveN(unassigned_obj, index=0, priority=2)
        
        # Priority 2: Minimize other penalties
        obj_terms = []

        # Time penalties
        opt_weights = self.reader.optimization
        time_weight = opt_weights.get('time', 0) if opt_weights else 0
        
        for cid in self.reader.classes.keys():
            time_opts = self.class_to_time_options[cid]
            for topt, tidx in time_opts:
                penalty = topt.get('penalty', 0)
                if penalty > 0:
                    obj_terms.append(time_weight * penalty * self.y[cid, tidx])
        
        # Distribution penalties
        dist_weight = opt_weights.get('distribution', 0) if opt_weights else 0
        for p_var, cost in self.penalty_vars_time:
            obj_terms.append(dist_weight * cost * p_var)
        
        # Unassigned penalty
        # obj_terms.append(gp.quicksum(self.u_time[cid] * BIG_M for cid in self.reader.classes.keys()))
        
        if obj_terms:
            penalties_obj = gp.quicksum(obj_terms)
            self.model_time.setObjective(penalties_obj, sense=GRB.MINIMIZE)
        # Set multi-objective parameters
        # self.model_time.ModelSense = GRB.MINIMIZE
    
    def _solve_time_step(self):
        """Solve time assignment step"""
        print("Solving time assignment...")
        start_time = time.time()
        
        self.model_time.optimize()
        solve_time = time.time() - start_time
        
        if self.model_time.Status == GRB.OPTIMAL or self.model_time.Status == GRB.SUBOPTIMAL:
            self.logger.info(f"✓ Time step optimal! Objective: {self.model_time.ObjVal:.2f}, Time: {solve_time:.2f}s")
        elif self.model_time.Status == GRB.TIME_LIMIT:
            self.logger.info(f"⚠ Time step reached time limit. Objective: {self.model_time.ObjVal:.2f}")
        elif self.model_time.Status == GRB.INFEASIBLE:
            self.logger.info(f"\n✗ Model is infeasible")
            self.model_time.computeIIS()
            self.model_time.write("infeasible.ilp")
            return False
        else:
            self.logger.info(f"✗ Time step failed with status {self.model_time.Status}")
            return False
    
        # Extract time assignments
        for i in range(self.model_time.SolCount):
            time_assignments = {}
            self.model_time.setParam('SolutionNumber', i)
            for cid in self.reader.classes.keys():
                time_opts = self.class_to_time_options[cid]
                for topt, tidx in time_opts:
                    if (cid, tidx) in self.y and self.y[cid, tidx].Xn > 0.5:
                        time_assignments[cid] = (topt, tidx)
                        break
                if time_assignments.get(cid, None) == None:
                    print(f"cid {cid} has no time assignment")
            self.time_assignments_results.append(time_assignments)
        self.time_assignments = self.time_assignments_results[0]
        assigned_count = len(self.time_assignments)
        total_count = len(self.reader.classes)
        self.logger.info(f"Time step: {assigned_count}/{total_count} classes assigned ({100*assigned_count/total_count:.1f}%)")
        
        return True
    
    def _build_room_model(self):
        """Build MIP model for room assignment given time assignments"""
        print("Building room assignment model...")
        
        # Step 2: Room assignment model
        self.model_room = gp.Model("RoomAssignment")
        self.model_room.setParam('TimeLimit', self.time_limit * 0.5)
        self.model_room.setParam('Threads', self.Threads)
        self.model_room.setParam('MIPGap', self.MIPGap)
        self.model_room.setParam('MIPFocus', 1)
        self.w = {}  # w[cid, room_id]: class-room assignment
        self.penalty_vars_room = []
        # self.u_room = {}  # unassigned in room step

        # Create variables only for assigned classes
        for cid in self.reader.classes.keys():
            room_options = self.class_to_room_options[cid]
            
            # self.u_room[cid] = self.model_room.addVar(
            #     vtype=GRB.BINARY,
            #     name=f"u_room_{cid}"
            # )
            for rid in room_options:
                self.w[cid, rid] = self.model_room.addVar(
                    vtype=GRB.BINARY,
                    name=f"w_{cid}_{rid}"
                )
                self.w[cid, rid].setAttr("BranchPriority", self.class_to_value[cid])
        
        # Constraints: each class assigned to exactly one room or unassigned
        for cid in self.reader.classes.keys():
            room_options = self.class_to_room_options[cid]
            w_vars = [self.w[cid, rid] for rid in room_options]
            
            self.model_room.addConstr(
                gp.quicksum(w_vars) == 1,
                # gp.quicksum(w_vars) + self.u_room[cid] == 1,
                name=f"assign_room_{cid}"
            )
        
        # Room capacity and unavailable constraints (no double booking and unavailable)
        self._add_room_self_constraints()

        # self._add_room_unavailable_constraints()
        
        # Room-related distribution constraints
        self._add_room_constraints()
        
        # Objective
        self._set_room_objective()
        
        print(f"Room model: {self.model_room.NumVars} vars, {self.model_room.NumConstrs} constrs")
    
    def _add_room_self_constraints(self):
        """Add room double-booking and unavailable prevention constraints"""
        print("Adding room capacity and unavailable constraints...")
        
        constraint_count = 0
        
        for rid in self.reader.rooms.keys():
            classes_using_room = []
            for cid in self.reader.classes.keys():
                room_opts = self.class_to_room_options[cid]
                if rid in room_opts:
                    classes_using_room.append(cid)
            
            if len(classes_using_room) < 2:
                continue
            
            for i, c1 in enumerate(classes_using_room):
                if c1 not in self.time_assignments:
                    continue
                topt1, tidx1 = self.time_assignments[c1]
                bits1 = topt1['optional_time_bits']
                for bits2 in self.reader.rooms[rid]['unavailables_bits']:
                    if self._times_conflict(bits1, bits2):
                        # Cannot both use this room
                        self.model_room.addConstr(
                            self.w[c1, rid] <= 0,
                            name=f"room_unavailable_{rid}_{c1}"
                        )
                        constraint_count += 1

            # For each pair of classes that might use this room
            for i, c1 in enumerate(classes_using_room):
                for c2 in classes_using_room[i+1:]:
                    # Check if their time slots conflict
                    if c1 not in self.time_assignments or c2 not in self.time_assignments:
                        continue
                    
                    topt1, tidx1 = self.time_assignments[c1]
                    topt2, tidx2 = self.time_assignments[c2]
                    bits1 = topt1['optional_time_bits']
                    bits2 = topt2['optional_time_bits']
                    
                    if self._times_conflict(bits1, bits2):
                        # Cannot both use this room
                        self.model_room.addConstr(
                            self.w[c1, rid] + self.w[c2, rid] <= 1,
                            name=f"room_cap_{rid}_{c1}_{c2}"
                        )
                        constraint_count += 1
        
        print(f"Added {constraint_count} room capacity constraints")
    
    def _add_room_constraints(self):
        """Add room-related distribution constraints"""
        print("Adding room distribution constraints...")
        
        for constraint in self.reader.distributions['hard_constraints']:
            self._add_room_constraint(constraint, is_hard=True)
        
        # for constraint in self.reader.distributions['soft_constraints']:
        #     self._add_room_constraint(constraint, is_hard=False)
    
    def _add_room_constraint(self, constraint, is_hard):
        """Add a single room-related constraint"""
        ctype = constraint['type']
        classes = constraint['classes']
        penalty = constraint.get('penalty', 0)
        
        if ctype == 'SameRoom':
            self._add_same_room_constraint(classes, is_hard, penalty)
        elif ctype == 'DifferentRoom':
            self._add_different_room_constraint(classes, is_hard, penalty)
        elif ctype == 'SameAttendees':
            # In room step, check travel times
            self._add_same_attendees_room_constraint(classes, is_hard, penalty)
    
    def _add_same_room_constraint(self, classes, is_hard, penalty):
        """SameRoom: courses must be in same room"""
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            rooms1 = self.class_to_room_options[c1]
            rooms2 = self.class_to_room_options[c2]
            common_rooms = set(rooms1) & set(rooms2)
            
            if is_hard:
                for r in common_rooms:
                    self.model_room.addConstr(
                        self.w[c1, r] == self.w[c2, r],
                        name=f"same_room_{c1}_{c2}_{r}"
                    )
            else:
                for r in common_rooms:
                    p = self.model_room.addVar(vtype=GRB.BINARY)
                    self.model_room.addConstr(self.w[c1, r] - self.w[c2, r] <= p)
                    self.model_room.addConstr(self.w[c2, r] - self.w[c1, r] <= p)
                    self.penalty_vars_room.append((p, penalty))
    
    def _add_different_room_constraint(self, classes, is_hard, penalty):
        """DifferentRoom: courses must be in different rooms"""
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            rooms1 = self.class_to_room_options[c1]
            rooms2 = self.class_to_room_options[c2]
            common_rooms = set(rooms1) & set(rooms2)
            
            for r in common_rooms:
                if is_hard:
                    self.model_room.addConstr(
                        self.w[c1, r] + self.w[c2, r] <= 1,
                        name=f"diff_room_{c1}_{c2}_{r}"
                    )
                else:
                    p = self.model_room.addVar(vtype=GRB.BINARY)
                    self.model_room.addConstr(
                        self.w[c1, r] + self.w[c2, r] - 1 <= p
                    )
                    self.penalty_vars_room.append((p, penalty))
    
    def _add_same_attendees_room_constraint(self, classes, is_hard, penalty):
        """SameAttendees: check travel time constraints between assigned times/rooms"""
        if len(classes) < 2:
            return
        
        travel_times = self.reader.travel if self.reader.travel else {}
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            if c1 not in self.time_assignments or c2 not in self.time_assignments:
                continue
            
            topt1, _ = self.time_assignments[c1]
            topt2, _ = self.time_assignments[c2]
            week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
            week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']
            days_int1 = int(day_bits1, 2)
            days_int2 = int(day_bits2, 2)
            and_days = days_int1 & days_int2
            week_int1 = int(week_bits1, 2)
            week_int2 = int(week_bits2, 2)
            and_week = week_int1 & week_int2
            
            rooms1 = self.class_to_room_options[c1]
            rooms2 = self.class_to_room_options[c2]
            
            for r1 in rooms1:
                for r2 in rooms2:
                    if r1 == 'dummy' or r2 == 'dummy':
                        continue
                    travel1 = travel_times.get(r1, {}).get(r2, 0)
                    travel2 = travel_times.get(r2, {}).get(r1, 0)
                    viot = False
                    if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0): # Overlap
                        viot = True
                    if (start1 + end1 + travel1 <= start2) or (start2 + end2 + travel2 <= start1) or (and_days == 0) or (and_week == 0):
                        continue
                    else:
                        viot = True
                    if viot:
                        if is_hard:
                            self.model_room.addConstr(
                                self.w[c1, r1] + self.w[c2, r2] <= 1,
                                name=f"travel_{c1}_{r1}_{c2}_{r2}"
                            )
                        else:
                            p = self.model_room.addVar(vtype=GRB.BINARY)
                            self.model_room.addConstr(
                                self.w[c1, r1] + self.w[c2, r2] - 1 <= p
                            )
                            self.penalty_vars_room.append((p, penalty))
    
    def _set_room_objective(self):
        """Set objective for room assignment step"""
        print("Setting room objective...")
        
        # Priority 1: Minimize unassigned room
        # unassigned_obj = gp.quicksum(self.u_room[cid] for cid in self.reader.classes.keys())
        # self.model_room.setObjectiveN(unassigned_obj, index=0, priority=10)

        # Priority 2: Minimize other penalties
        obj_terms = []

        # Room penalties
        opt_weights = self.reader.optimization
        room_weight = opt_weights.get('room', 0) if opt_weights else 0
        
        for cid in self.reader.classes.keys():
            class_data = self.reader.classes[cid]
            for ropt in class_data['room_options']:
                rid = ropt['id']
                penalty = ropt.get('penalty', 0)
                if penalty > 0 and (cid, rid) in self.w:
                    obj_terms.append(room_weight * penalty * self.w[cid, rid])
        
        # Distribution penalties
        dist_weight = opt_weights.get('distribution', 0) if opt_weights else 0
        for p_var, cost in self.penalty_vars_room:
            obj_terms.append(dist_weight * cost * p_var)
        
        # Unassigned penalty
        # obj_terms.append(gp.quicksum(self.u_room[cid] * BIG_M for cid in self.reader.classes.keys()))
        
        if obj_terms:
            penalties_obj = gp.quicksum(obj_terms)
            self.model_room.setObjective(penalties_obj, sense=GRB.MINIMIZE)
            # self.model_room.setObjectiveN(penalties_obj, index=1, priority=1)
            print(f"✓ 房间目标函数设置完成，包含 {len(obj_terms)} 项")
        # self.model_room.ModelSense = GRB.MINIMIZE
    
    # def next_solution(self, it):
    #     self.logger.info(f"set solution to the {it} optimal")
    #     self.model_time.setParam('SolutionNumber', it)
    #     self.time_assignments = {}
    #     try:
    #         all_vars
    #     for cid in self.reader.classes.keys():
    #         time_opts = self.class_to_time_options[cid]
    #         for topt, tidx in time_opts:
    #             if (cid, tidx) in self.y and self.y[cid, tidx].Xn > 0.5:
    #                 self.time_assignments[cid] = (topt, tidx)
    #                 break
    #         if self.time_assignments.get(cid, None) == None:
    #             print(f"cid {cid} has no time assignment")

    def next_solution(self, it):
        self.logger.info(f"Setting solution to the {it} index")
        self.time_assignments = self.time_assignments_results[it]
        

    def add_room_conflict_cut(self):
        for it, sol_data in enumerate(self.time_assignments_results):
            ones = []
            zeros = []
            
            # 这里的关键是：利用你预存的数据来区分哪些变量该在约束的哪一侧
            # 假设 sol_data 存储的是 (cid, tidx) 的 key 集合
            for (cid, tidx), var in self.y.items():
                if (cid, tidx) in sol_data: # 这里的 sol_data 是该解中所有为 1 的坐标集合
                    ones.append(var)
                else:
                    zeros.append(var)
            self.model_time.addConstr(
                gp.quicksum(1 - v for v in ones) + 
                gp.quicksum(v for v in zeros) >= 1,
                name=f"cutoff_failure_config_it{it}"
            )
        self.time_assignments = {}
        self.room_assignments = {}
        self.model_time.update()

    def _solve_room_step(self):
        """Solve room assignment step"""
        print("Solving room assignment...")
        start_time = time.time()
        
        self.model_room.optimize()
        solve_time = time.time() - start_time
        
        if self.model_room.Status == GRB.OPTIMAL or self.model_room.Status == GRB.SUBOPTIMAL:
            self.logger.info(f"✓ Room step optimal! Objective: {self.model_room.ObjVal:.2f}, Time: {solve_time:.2f}s")
        elif self.model_room.Status == GRB.TIME_LIMIT:
            self.logger.info(f"⚠ Room step reached time limit. Objective: {self.model_room.ObjVal:.2f}")
        elif self.model_room.Status == GRB.INFEASIBLE:
            self.logger.info(f"\n✗ Model is infeasible")
            return False
        else:
            self.logger.info(f"✗ Room step failed with status {self.model_room.Status}")
            return False
        
        # Extract assignments
        assigned_count = 0
        for i in range(self.model_room.SolCount):
            room_assignments = {}
            self.model_room.setParam('SolutionNumber', i)
            for cid in self.reader.classes.keys():
                room_opts = self.class_to_room_options[cid]
                for rid in room_opts:
                    if (cid, rid) in self.w and self.w[cid, rid].Xn > 0.5:
                        room_assignments[cid] = rid
                        break
            if len(room_assignments) >= assigned_count:
                assigned_count = len(room_assignments)
                self.room_assignments = room_assignments
        
        total_count = len(self.reader.classes)
        self.logger.info(f"Room step: {assigned_count}/{total_count} classes assigned ({100*assigned_count/total_count:.1f}%)")

        return True
    
    def extract_solution(self):
        """Extract final solution"""
        if self.model_room.SolCount == 0:
            return None
        
        print("\n=== Extracting Solution ===")
        
        assignments = {}
        
        for cid in self.reader.classes.keys():
            class_data = self.reader.classes[cid]
            
            assigned_time = None
            assigned_room = None
            
            if cid in self.time_assignments:
                assigned_time, _ = self.time_assignments[cid]
            
            room_opts = self.class_to_room_options[cid]
            for rid in room_opts:
                if (cid, rid) in self.w and self.w[cid, rid].X > 0.5:
                    assigned_room = rid
                    break
            
            room_required = class_data.get('room_required', True)
            if assigned_room == 'dummy':
                assigned_room = None
            
            assignments[cid] = (assigned_time, room_required, assigned_room, [])
            
            # if assigned_time:
            #     bits = assigned_time['optional_time_bits']
            #     print(f"  Class {cid}: weeks={bits[0][:8]}... days={bits[1]} "
            #         f"start={bits[2]} length={bits[3]} room={assigned_room}")
        
        assigned_count = 0
        for cid, assignment in assignments.items():
            if (assignment[0] is not None) and ((not assignment[1]) or (assignment[2] is not None)):
                assigned_count += 1
        total_count = len(assignments)
        self.logger.info(f"\nTotal: {assigned_count}/{total_count} classes assigned ({100*assigned_count/total_count:.1f}%)")
        
        return assignments
    
    def save_solution(self, assignments, output_path, config):
        """Save solution to XML"""
        if assignments is None:
            print("No solution to save")
            return
        
        total_time = self.model_time.Runtime + self.model_room.Runtime
        
        export_solution_xml(
            assignments=assignments,
            out_path=output_path,
            name=self.reader.problem_name,
            runtime_sec=total_time,
            cores=self.Threads,
            technique=config['config']['technique'],
            author=config['config']['author'],
            institution=config['config']['institution'],
            country=config['config']['country'],
            include_students=config['config']['include_students']
        )
        
        self.logger.info(f"\n✓ Solution saved to: {output_path}")