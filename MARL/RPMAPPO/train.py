import time
from math import inf
from tqdm import tqdm
import random
import numpy as np
from MARL.RPMAPPO.env import CustomEnvironment
from MARL.RPMAPPO.Scheduler import Scheduler
from MARL.RPMAPPO.MAPPO import MAPPO

def train(reader, logger, tools, output_folder, fileName, config, quickrun=False):
    pname = fileName.split('.xml')[0]

    update_metrics = tools.update_metrics
    conclude = tools.conclude
    metrics_list = ["episode_lengths", "scheduler epsilon", "unassigned room", "Time penalty", "Room penalty", "Distribution penalty", "mappo policy loss", "mappo value loss", "mappo entropies", "Total cost", "Avg Scheduler Reward", "Avg Mappo Reward"]
    tools.set_metrics(metrics_list)

    report = config['config']['report']
    steps_clip = config['train']['steps_clip']
    random_warmup = config['train']['random_warmup']
    warmup_episode = config['train']['warmup_episode']
    total_episodes = int(config['train']['total_episodes'])

    logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    env = CustomEnvironment(reader, config)
    mappo_obs, masks, sched_obs, none_assignment = env.reset(order=True)
    team_size = len(env.agents)
    state_dim = len(env.agents)
    action_dim = env.max_value
    sched_obs_dim = len(sched_obs)
    sched_action_dim = len(sched_obs)
    scheduler = Scheduler(sched_obs_dim, sched_action_dim, config)
    mappo = MAPPO(team_size, state_dim, action_dim, config)
    sched_mask = [1 for _ in sched_obs]
    warm_up = True
    fail = 0
    last_sched_reward = -inf
    t0 = time.perf_counter()

    for episode in tqdm(range(total_episodes), desc=f"Training"):
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
        } for _ in range(team_size)]
        sched_reward = []
        sched_cost = inf
        best_iter = steps_clip
        sched_none_assignment_num = sched_obs_dim
        best_result = {}
        
        t0_ep = time.perf_counter()
        if warm_up and episode < warmup_episode:
            warm_up_iters = 0
            env.warm_up = True
            best_warm_up_cost = inf
            mappo_obs, masks, sched_obs, none_assignment = env.reset(order=True)
            sched_mask = [1 for _ in sched_obs]
            random_pbar = tqdm(range(random_warmup))
            for step in random_pbar:
                warm_up_iters += 1
                random_pbar.set_description(f"warm_up_iters {warm_up_iters} current {len(none_assignment)}/{sched_obs_dim} unassigned")
                result = env.step()
                none_assignment = result['not assignment']
                if len(none_assignment) == 0:
                    random_pbar.set_description(f"warm_up_iters {warm_up_iters} current {len(none_assignment)}/{sched_obs_dim} unassigned")
                    sched_obs = result['scheduler_observations']
                    sched_mask = result['scheduler_mask']
                    sched_reward = result['rewards']
                    sched_cost = result["Total cost"]
                    best_result = {
                        "Total cost": result["Total cost"],
                        "Time penalty": result["Time penalty"], 
                        "Room penalty": result["Room penalty"], 
                        "Distribution penalty": result["Distribution penalty"],
                    }
                    break
                elif best_warm_up_cost > result["Total cost"]:
                    best_warm_up_cost = result["Total cost"]
                    sched_obs = result['scheduler_observations']
                    sched_mask = result['scheduler_mask']
                    sched_reward = result['rewards']
                    sched_cost = result["Total cost"]
                _, _ = env.reset_step()
                env.order_agents()
            env.warm_up = False
            sched_none_assignment_num = len(none_assignment)

        iters = 0
        epsilon = scheduler.set_epsilon(episode)
        sched_actions = scheduler.take_action(sched_obs, sched_mask)
        env.apply_scheduling(sched_obs, sched_mask, sched_actions)

        sched_buffers['states'] = sched_obs
        sched_buffers['actions'] = sched_actions
        
        best_iter = steps_clip
        Avg_mappo_reward = []

        pbar = tqdm(range(steps_clip))
        for step in pbar:
            pbar.set_description(f"iters {iters} best result {sched_none_assignment_num}/{sched_obs_dim} unassigned")
            iters += 1
            mappo_obs, masks = env.reset_step()
            state_list = [mappo_obs[agent.id].flatten() for agent in env.agents]
            mask_list = [masks[agent.id].flatten() for agent in env.agents]
            probs = mappo.take_action(state_list, mask_list)
            env.apply_mappo_action(probs)
            result = env.step()
            none_assignment = result['not assignment']
            next_mappo_obs = result['mappo_observations']
            next_states = result['scheduler_observations']
            _sched_mask = result['scheduler_mask']
            rewards = result['rewards']
            mappo_actions = result['actions']
            mappo_mask = result['masked_actions']
            Avg_mappo_reward.append(np.mean(rewards))
            for i, agent in enumerate(env.agents):
                mappo_buffers[i]['states'].append(np.array(mappo_obs[agent.id]))
                mappo_buffers[i]['actions'].append(mappo_actions[agent.id])
                mappo_buffers[i]['mask_actions'].append(mappo_mask[agent.id])
                mappo_buffers[i]['next_states'].append(np.array(next_mappo_obs[agent.id]))
                mappo_buffers[i]['rewards'].append(float(rewards[i]))
                mappo_buffers[i]['dones'].append(float(agent not in none_assignment))
                mappo_buffers[i]['action_probs'].append(probs[i])
            if sched_none_assignment_num > len(none_assignment) or (sched_none_assignment_num == len(none_assignment) and sched_cost > result["Total cost"]):
                sched_obs = next_states
                sched_reward = rewards
                sched_mask = _sched_mask
                sched_none_assignment_num = len(none_assignment)
                sched_cost = result["Total cost"]
                best_result = {
                    "Total cost": result["Total cost"],
                    "Time penalty": result["Time penalty"], 
                    "Room penalty": result["Room penalty"], 
                    "Distribution penalty": result["Distribution penalty"],
                }
            if sched_none_assignment_num == 0 and sched_cost > result["Total cost"]:
                best_iter = iters
            mappo_obs = next_mappo_obs
        runtime = time.perf_counter() - t0
        a_loss, c_loss, ent = mappo.update(mappo_buffers)

        sched_buffers['rewards'] = sched_reward
        sched_buffers['next_states'] = sched_obs
        scheduler.update(sched_buffers)
        
        avg_sched_reward = np.mean(sched_reward)

        metrics = {
            "episode_lengths": best_iter,
            "scheduler epsilon": epsilon,
            "unassigned room": sched_none_assignment_num,
            "mappo policy loss": a_loss,
            "mappo value loss": c_loss,
            "mappo entropies": ent,
            "Avg Scheduler Reward": avg_sched_reward,
            "Avg Mappo Reward": np.mean(Avg_mappo_reward)
        }
        if sched_none_assignment_num != 0: 
            warm_up = False
            metrics.update(result)
            if avg_sched_reward < last_sched_reward:
                fail += 1
            last_sched_reward = avg_sched_reward
            if fail >= 3:
                warm_up = True
                fail = 0
                last_sched_reward = -inf
            else:
                _, _, _, none_assignment = env.reset()
        else: 
            warm_up = True
            metrics.update(best_result)
        isbest = update_metrics(metrics, sched_none_assignment_num, env, pname, output_folder, runtime)
        if isbest: 
            scheduler.save(pname)
            mappo.save(pname)
        if isbest and quickrun:
            break
        if episode >= warmup_episode:
            if avg_sched_reward < last_sched_reward:
                fail += 1
            last_sched_reward = avg_sched_reward
            if fail >= 3:
                mappo_obs, masks, sched_obs, none_assignment = env.reset(order=True)
                sched_mask = [1 for _ in sched_obs]
                fail = 0
                last_sched_reward = -inf
            else:
                _, _, _, none_assignment = env.reset()
            

    runtime = time.perf_counter() - t0
    conclude(runtime, report, pname, output_folder)
