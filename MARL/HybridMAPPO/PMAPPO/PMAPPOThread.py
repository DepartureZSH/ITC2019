import threading
import logging
from MARL.HybridMAPPO.PMAPPO.env import CustomEnvironment
from MARL.HybridMAPPO.PMAPPO.Scheduler import Scheduler
from MARL.HybridMAPPO.PMAPPO.MAPPO import MAPPO
from queue import Queue
import queue
from math import inf
from tqdm import tqdm
import numpy as np
import time

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

class PMAPPOThread(threading.Thread):
    def __init__(self, solution_queue, tools, output_folder, pname, reader, config, quickrun=False):
        super().__init__()
        self.solver_id = "PMAPPO"
        self.solution_queue = solution_queue
        self.output_folder = output_folder
        self.pname = pname
        setup_logger(self.solver_id, f"{self.output_folder}/logger.log")
        self.logger = logging.getLogger(self.solver_id)
        self.logger.info(f"PMAPPO logger set to {self.output_folder}/logger.log")
        self.daemon = True
        self.is_running = True

        self.model_init(tools, reader, config)
    
    def model_init(self, tools, reader, config):
        self.reader = reader
        self.config = config
        self.env = CustomEnvironment(reader, config)
        mappo_obs, masks, sched_obs, none_assignment = self.env.reset(order=True)
        self.team_size = len(self.env.agents)
        self.state_dim = len(self.env.agents)
        self.action_dim = self.env.max_value
        sched_obs_dim = len(sched_obs)
        sched_action_dim = len(sched_obs)
        self.scheduler = Scheduler(sched_obs_dim, sched_action_dim, config, self.output_folder)
        self.mappo = MAPPO(self.team_size, self.state_dim, self.action_dim, config, self.output_folder)
        self.update_metrics = tools.update_metrics
        self.conclude = tools.conclude
        self.total_episodes = int(self.config['train']['total_episodes'])
        self.steps_clip = config['train']['steps_clip']
        self.report = config['config']['report']
        metrics_list = ["Valid Soluttion", "Avg Valid Total Cost", "Avg Mappo Reward", "Total Cost Benchmark", "Best Total Cost", "Optimization Degree", "Avg Unassigned Room", "Scheduler Epsilon", "Avg Scheduler Reward", "Mappo policy loss", "Mappo value loss", "Mappo entropies"]
        tools.set_metrics(metrics_list)
        
    def run(self):
        self.logger.info(f"Optimizer-{self.solver_id} start, waiting for valid solution...")
        Total_cost = []
        t0 = time.perf_counter()
        epsilon = self.scheduler.max_epsilon
        iters = 0
        while self.is_running:
            try:
                solution = self.solution_queue.get(timeout=1)
                if solution is None:
                    self.solution_queue.task_done()
                    break
                solution_path = solution['solution']
                value_path = solution.get('value', None)
                self.logger.info("##################################################################")
                self.logger.info(f"Optimizer start to optimize: {solution_path.split('/')[-1]} with value file {value_path.split('/')[-1]}")
                ##################################################################
                sched_buffers = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                }
                mappo_buffers = [{
                    'states': [], 
                    'actions': [],
                    'mask_actions': [],
                    'next_states': [], 
                    'rewards': [], 
                    'dones': [], 
                    'action_probs': [],
                } for _ in range(self.team_size)]
                _, _, _, none_assignment = self.env.reset(order=True)
                mappo_obs, sched_obs, mappo_masks, initial_actions, quality = self.env.initial_solution(solution_path, value_path)
                sched_mask = [1 for _ in sched_obs]
                ##################################################################
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
                avg_sched_reward_per = []
                episode_costs  =[quality['Total cost']]
                episode_unassigned = [0]
                episode_valid_solution = 0
                valid_actions = [initial_actions]

                CostBenchmark = quality['Total cost']

                BestCost = quality['Total cost']
                BestTimeCost = quality['Time penalty']
                BestRoomCost = quality['Room penalty']
                BestDistributionCost = quality['Distribution penalty']

                # pbar = tqdm(range(self.total_episodes))
                for episode in range(self.total_episodes):
                    epsilon = self.scheduler.set_epsilon(iters)
                    sched_actions = self.scheduler.take_action(sched_obs, sched_mask)
                    
                    self.env.apply_scheduling(sched_obs, sched_mask, sched_actions)

                    sched_buffers['states'] = sched_obs
                    sched_buffers['actions'] = sched_actions
                    sched_rewards = []
                    sched_reward = []
                    sched_none_assignment_num = len(none_assignment)
                    sched_cost = inf

                    # pbar.set_description(f"iters {iters}")
                    for step in range(self.steps_clip):
                        iters += 1
                        state_list = [mappo_obs[agent.id].flatten() for agent in self.env.agents]
                        mask_list = [mappo_masks[agent.id].flatten() for agent in self.env.agents]
                        probs = self.mappo.take_action(state_list, mask_list)
                        self.env.apply_mappo_action(probs)
                        result = self.env.step()
                        none_assignment = result['not assignment']
                        next_mappo_obs = result['mappo_observations']
                        next_states = result['scheduler_observations']
                        _sched_mask = result['scheduler_mask']
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
                            valid = True
                            for valid_action in valid_actions:
                                if valid_action == result['actions']:
                                    print("Solution duplicated")
                                    valid = False
                                    break
                            if valid:
                                episode_valid_solution += 1
                                sched_obs = next_states
                                sched_rewards.append(rewards)
                                sched_mask = _sched_mask
                                valid_actions.append(result['actions'])
                                if BestCost > result["Total cost"]:
                                    BestCost = result["Total cost"]
                                    BestTimeCost = result["Time penalty"]
                                    BestRoomCost = result["Room penalty"]
                                    BestDistributionCost = result["Distribution penalty"]
                                episode_costs.append(result["Total cost"])
                        if len(sched_rewards) == 0:
                            sched_obs = next_states
                            sched_rewards.append(rewards)
                            sched_mask = _sched_mask
                        episode_unassigned.append(len(none_assignment))
                        mappo_obs = next_mappo_obs
                        mappo_masks = self.env.reset_step()
                    runtime = time.perf_counter() - t0
                    a_loss, c_loss, ent = self.mappo.update(mappo_buffers)
                    sched_reward = sched_rewards[0]
                    for j in range(1, len(sched_rewards)):
                        reward = sched_rewards[j]
                        for i, agent in enumerate(self.env.agents):
                            sched_reward[i] += float(reward[i])
                    # for i, agent in enumerate(self.env.agents):
                    #     sched_reward[i] = sched_reward[i] / len(sched_rewards)
                    sched_buffers['rewards'] = sched_reward
                    avg_sched_reward_per.append(np.mean(sched_reward))
                    sched_buffers['next_states'] = sched_obs
                    self.scheduler.update(sched_buffers)
                avg_sched_reward = np.mean(avg_sched_reward_per)
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
                    "Avg Unassigned Room": float(np.mean(episode_unassigned)),
                    "Scheduler Epsilon": epsilon,
                    "Avg Scheduler Reward": avg_sched_reward,
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

        