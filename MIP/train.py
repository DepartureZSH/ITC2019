import time
from math import inf
from tqdm import tqdm
import random
import numpy as np
from MIP.solver import MIPSolver
from MIP.two_solver import TwoStepSolverTimeFirst

def MIP2Step_Solver(reader, logger, tools, fileName, config):
    output_folder = config['config']['output']
    pname = fileName.split('.xml')[0]

    logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    solver = MIPSolver(reader, logger, config)

    # 构建模型
    if config['method'].get('reproduction', False):
        model_path = f'{output_folder}/{pname}/{pname}'
        solver.load_model(model_path)
    else:
        solver.build_model()

    # 求解
    epoch = int(config['method']['epoch'])
    c = 1
    for i in range(epoch):
        assignments_list = solver.solve()

        if len(assignments_list) > 0:
            output_path = f'{output_folder}/{pname}/{pname}'
            solver.save_model(output_path)

        for assignments in assignments_list:
            output_file = f'{output_folder}/{pname}/solution{c}_{pname}.xml'
            c += 1
            if c >= 100:
                logger.info("====1000 solution====")
                return
            solver.save_solution(assignments, output_file, config)
        

def MIP3Step_Solver(reader, logger, tools, fileName, config):
    output_folder = config['config']['output']
    pname = fileName.split('.xml')[0]

    output_file = f'{output_folder}/solution_{pname}.xml'

    logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    solver = TwoStepSolverTimeFirst(reader, logger, config)

    # 构建并求解模型
    assignments = solver.build_and_solve()

    # 保存解决方案
    if assignments:
        solver.save_solution(assignments, output_file, config)
