import threading
import logging
from MARL.HybridMAPPO.MAPPO.env import CustomEnvironment
from MARL.HybridMAPPO.MAPPO.MAPPO import MAPPO
from queue import Queue
import queue
from math import inf
from tqdm import tqdm
import numpy as np
import time
import random

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler) 

class MAPPOThread(threading.Thread):
    def __init__(self, solution_queue, tools, output_folder, pname, reader, config, quickrun=False):
        super().__init__()
        self.solver_id = "MAPPO"
        self.solution_queue = solution_queue
        self.output_folder = output_folder
        self.pname = pname
        setup_logger(self.solver_id, f"{self.output_folder}/logger.log")
        self.logger = logging.getLogger(self.solver_id)
        self.logger.info(f"MAPPO logger set to {self.output_folder}/logger.log")
        self.daemon = True
        self.is_running = True

        self.model_init(tools, reader, config)
    
    def model_init(self, tools, reader, config):
        self.reader = reader
        self.config = config
        self.env = CustomEnvironment(reader, config)
        mappo_obs, masks, none_assignment = self.env.reset(order=True)
        self.team_size = len(self.env.agents)
        self.state_dim = len(self.env.agents)
        self.action_dim = self.env.max_value
        self.mappo = MAPPO(self.team_size, self.state_dim, self.action_dim, config, self.output_folder)
        self.update_metrics = tools.update_metrics
        self.conclude = tools.conclude
        self.total_episodes = int(self.config['train']['total_episodes'])
        self.steps_clip = config['train']['steps_clip']
        self.report = config['config']['report']
        metrics_list = ["Valid Soluttion", "Avg Valid Total Cost", "Avg Mappo Reward", "Total Cost Benchmark", "Best Total Cost", "Optimization Degree", "Best Time Penalty", "Best Room Penalty", "Best Distribution Penalty", "Mappo policy loss", "Mappo value loss", "Mappo entropies"]
        tools.set_metrics(metrics_list)
        
    def run(self):
        self.logger.info(f"Optimizer-{self.solver_id} start, waiting for valid solution...")
        mappo_buffers = [{
            'states': [], 
            'actions': [],
            'mask_actions': [],
            'next_states': [], 
            'rewards': [], 
            'dones': [], 
            'action_probs': [],
        } for _ in range(self.team_size)]
        Total_cost = []
        t0 = time.perf_counter()
        while self.is_running:
            try:
                solution = self.solution_queue.get(timeout=1)
                if solution is None:
                    self.solution_queue.task_done()
                    break
                solution_path = solution['solution']
                value_path = solution['value']
                self.logger.info("##################################################################")
                self.logger.info(f"Optimizer start to optimize: {solution_path.split('/')[-1]}")
                ##################################################################
                _, _, none_assignment = self.env.reset(order=True)
                mappo_obs, masks, quality = self.env.initial_solution(solution_path, value_path)
                self.logger.info(f"initial solution quality:")
                if len(quality['not assignment']) == 0:
                    self.logger.info(f"Valid solution: True")
                else:
                    self.logger.info(f"Valid solution: False")
                    self.logger.info(f"{len(quality['not assignment'])}/{len(self.env.agents)} class no vaild assignment")
                self.logger.info(f"Total cost: {quality['Total cost']}")
                Total_cost.append(quality['Total cost'])
                self.logger.info(f"Time penalty: {quality['Time penalty']}")
                self.logger.info(f"Room penalty: {quality['Room penalty']}")
                self.logger.info(f"Distribution penalty: {quality['Distribution penalty']}")
                ##################################################################
                t0_ep = time.perf_counter()
                Avg_mappo_reward = []
                episode_costs  =[]
                episode_unassigned = []
                episode_valid_solution = 0
                
                CostBenchmark = quality['Total cost']

                BestCost = quality['Total cost']
                BestTimeCost = quality['Time penalty']
                BestRoomCost = quality['Room penalty']
                BestDistributionCost = quality['Distribution penalty']

                # pbar = tqdm(range(self.total_episodes))
                iters = 0
                for episode in range(self.total_episodes):
                    # pbar.set_description(f"iters {iters}")
                    iters += 1
                    state_list = [mappo_obs[agent.id].flatten() for agent in self.env.agents]
                    mask_list = [masks[agent.id].flatten() for agent in self.env.agents]
                    probs = self.mappo.take_action(state_list, mask_list)
                    self.env.apply_mappo_action(probs)
                    result = self.env.step()
                    none_assignment = result['not assignment']
                    next_mappo_obs = result['mappo_observations']
                    rewards = result['rewards']
                    mappo_actions = result['actions']
                    mappo_mask = result['masked_actions']
                    Avg_mappo_reward.append(np.mean(rewards))
                    for i, agent in enumerate(self.env.agents):
                        mappo_buffers[i]['states'].append(np.array(mappo_obs[agent.id]))
                        mappo_buffers[i]['actions'].append(mappo_actions[agent.id])
                        mappo_buffers[i]['mask_actions'].append(mappo_mask[agent.id])
                        mappo_buffers[i]['next_states'].append(np.array(next_mappo_obs[agent.id]))
                        mappo_buffers[i]['rewards'].append(float(rewards[i]))
                        mappo_buffers[i]['dones'].append(float(agent not in none_assignment))
                        mappo_buffers[i]['action_probs'].append(probs[i])
                    if len(none_assignment) == 0:
                        episode_valid_solution += 1
                        if BestCost > result["Total cost"]:
                            BestCost = result["Total cost"]
                            BestTimeCost = result["Time penalty"]
                            BestRoomCost = result["Room penalty"]
                            BestDistributionCost = result["Distribution penalty"]
                        episode_costs.append(result["Total cost"])
                    episode_unassigned.append(len(none_assignment))
                    mappo_obs = next_mappo_obs
                    masks = self.env.reset_step()
                runtime = time.perf_counter() - t0
                a_loss, c_loss, ent = self.mappo.update(mappo_buffers)

                metrics = {
                    "Valid Soluttion": episode_valid_solution,
                    "Avg Valid Total Cost": float(np.mean(episode_costs)),
                    "Avg Mappo Reward": float(np.mean(Avg_mappo_reward)),
                    "Total Cost Benchmark": CostBenchmark,
                    "Best Total Cost": BestCost,
                    "Optimization Degree": (CostBenchmark - BestCost) / CostBenchmark * 100,
                    "Best Time Penalty": BestTimeCost,
                    "Best Room Penalty": BestRoomCost,
                    "Best Distribution Penalty": BestDistributionCost,
                    "Mappo policy loss": float(a_loss),
                    "Mappo value loss": float(c_loss),
                    "Mappo entropies": float(ent)
                }
                ##################################################################
                self.solution_queue.task_done()
                self.logger.info("##################################################################")
                self.logger.info(f"Optimizer complete optimize: {solution_path.split('/')[-1]}")
                self.logger.info(f"Solution quality after optimization:")
                if episode_valid_solution == 0:
                    self.logger.info(f"No valid solution found!")
                    self.logger.info(f"The average unassigned classes: {np.mean(episode_unassigned)}")
                else:
                    self.logger.info(f"{episode_valid_solution} valid solution found!")
                    self.logger.info(f"Optimization Degree: -{(CostBenchmark - BestCost) / CostBenchmark * 100} %")
                    self.logger.info(f"Best Total Cost: {BestCost}")
                    self.logger.info(f"Best Time Penalty: {BestTimeCost}")
                    self.logger.info(f"Best Room Penalty: {BestRoomCost}")
                    self.logger.info(f"Best Distribution Penalty: {BestDistributionCost}")
                if episode_valid_solution > 0:
                    isbest = self.update_metrics(metrics, 0, self.env, self.pname, self.output_folder, runtime)
                    if isbest:
                        self.mappo.save(self.output_folder)
                else:
                    isbest = self.update_metrics(metrics, np.mean(episode_unassigned), self.env, self.pname, self.output_folder, runtime)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.info(f"Optimizer error: {e}")
                self.solution_queue.task_done()
        runtime = time.perf_counter() - t0
        self.logger.info("##################################################################")
        self.conclude(runtime, self.report, self.pname, self.output_folder)
        min_cost = min(Total_cost)
        self.logger.info(f"Minial Total cost: {min_cost}", )
        self.logger.info("Optimizer exit")

        