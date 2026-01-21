import time
from math import inf
from tqdm import tqdm
import random
import numpy as np
import os
import threading
from queue import Queue
from MARL.HybridMAPPO.MAPPO.MAPPOThread import MAPPOThread
from MARL.HybridMAPPO.PMAPPO.PMAPPOThread import PMAPPOThread
from MARL.HybridMAPPO.MIP.MIPThread import MIPThread
from MARL.HybridMAPPO.Optimizer.env import CustomEnvironment
from MARL.HybridMAPPO.Optimizer.Scheduler import Scheduler
from MARL.HybridMAPPO.Optimizer.MAPPO import MAPPO

def check_solution_files_advanced(folder_path):
    if not os.path.isdir(folder_path):
        print(f"folder {folder_path} does not exist!")
        return False
    
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.startswith("solution") and entry.name.endswith('.xml'):
                return True

def get_solutions(folder_path):
    solutions = []
    value = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.startswith("solution") and entry.name.endswith('.xml'):
                solutions.append(f"{folder_path}/{entry.name}")
    if os.path.isdir(f"{folder_path}/model"):
        with os.scandir(f"{folder_path}/model") as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.json'):
                    value = f"{folder_path}/model/{entry.name}"
    return solutions, value

def make_folder(config, pname, dir):
    os.makedirs(f"{config['config']['output']}/{pname}/{dir}", exist_ok=True)
    return f"{config['config']['output']}/{pname}/{dir}"

def train(reader, logger, tools, output_folder, fileName, config, quickrun=False):
    isCGCS = config['train']['CGCS']['active']
    isRandom = config['train']['Random']['active']
    isMIP = config['train']['MIP']['active']
    isHistory = False
    isMAPPO = config['train']['MAPPO']['active']
    isPMAPPO = config['train']['PMAPPO']['active']

    pname = fileName.split('.xml')[0]
    output_folder_CGCS = make_folder(config, pname, "CGCS")
    output_folder_Random = make_folder(config, pname, "Random")
    output_folder_MIP = make_folder(config, pname, "MIP")
    output_folder_MAPPO = make_folder(config, pname, "MAPPO")
    output_folder_PMAPPO = make_folder(config, pname, "PMAPPO")

    SOLVER_NUM = 0
    QUEUE_MAX_SIZE = config['train']['QUEUE_MAX_SIZE']
    if isCGCS:
        SOLVER_NUM += 1
        logger.info("Solver implementation: CGCS")
    if isRandom:
        SOLVER_NUM += 1
        logger.info("Solver implementation: Random")
    if isMIP:
        SOLVER_NUM += 1
        logger.info("Solver implementation: MIP")
    if (not isCGCS) or (not isRandom) or (not isMIP):
        SOLVER_NUM = 1
        isHistory = True

    solution_queue = Queue(maxsize=QUEUE_MAX_SIZE)

    if isMAPPO:
        logger.info("Optimzer implementation: MAPPO")
    if isPMAPPO:
        logger.info("Optimzer implementation: PMAPPO")
    if isMAPPO and isPMAPPO:
        raise ValueError("Please specify only one Optimzer")

    logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    
    solver_threads = []
    #########################################################################################################
    # Solver: MIP
    # Input: reader, logger, config
    # Save: model, params
    # Output: solution
    # Output_path: output_folder/MIP
    #########################################################################################################
    if isMIP:
        MIPSolver = MIPThread(solution_queue, output_folder_MIP, pname, reader, config)
        solver_threads.append(MIPSolver)
        MIPSolver.start()

    #########################################################################################################
    # Solver: None
    # Input: 
    # Output: history solution
    #########################################################################################################
    if isHistory:
        Solutions = []
        if check_solution_files_advanced(output_folder_CGCS):
            logger.info(f"Use history solution from folder: {output_folder_CGCS}")
            solution, value = get_solutions(output_folder_CGCS)
            for solu in solution:
                Solutions.append({"solution": solu, "value": value})
        if check_solution_files_advanced(output_folder_Random):
            logger.info(f"Use history solution from folder: {output_folder_Random}")
            solution, value = get_solutions(output_folder_Random)
            for solu in solution:
                Solutions.append({"solution": solu, "value": value})
            # Solutions.extend(get_solutions(output_folder_Random))
        if check_solution_files_advanced(output_folder_MIP):
            logger.info(f"Use history solution from folder: {output_folder_MIP}")
            solution, value = get_solutions(output_folder_MIP)
            logger.info(f"{len(solution)} solutions found")
            for solu in solution:
                Solutions.append({"solution": solu, "value": value})
            # Solutions.extend(solution_files)
        if len(Solutions) == 0:
            raise ValueError("Please specify at least one Solver")
        for solution in Solutions:
            # print(solution)
            solution_queue.put(solution)
        logger.info(f"{len(Solutions)} solutions add to solution_queue")
    
    #########################################################################################################
    # Optimzer: MAPPO
    # Input: reader, solution
    # Save: model, params
    # Output: solution
    # Output_path: output_folder/MAPPO
    #########################################################################################################
    if isMAPPO:
        Optimizer = MAPPOThread(solution_queue, tools, output_folder_MAPPO, pname, reader, config, quickrun)
        Optimizer.start()

    #########################################################################################################
    # Optimzer: PMAPPO
    # Input: reader, solution
    # Save: model, params
    # Output: solution
    # Output_path: output_folder/PMAPPO
    #########################################################################################################
    if isPMAPPO:
        Optimizer = PMAPPOThread(solution_queue, tools, output_folder_PMAPPO, pname, reader, config, quickrun)
        Optimizer.start()

    for solver in solver_threads:
        solver.join()
    
    logger.info("All solver threads completed")

    solution_queue.put(None)

    Optimizer.join()
    solution_queue.join()

    logger.info("All valid Solution completed optimize. Program exit...")