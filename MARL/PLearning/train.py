import time
from tqdm import tqdm
import random
from MARL.PLearning.env import CustomEnvironment
from MARL.PLearning.network import Scheduler

def train(reader, logger, tools, output_folder, fileName, config, quickrun=False):
    pname = fileName.split('.xml')[0]

    update_metrics = tools.update_metrics
    conclude = tools.conclude
    metrics_list = ["episode_lengths", "runtime", "Total cost", "Time penalty", "Room penalty", "Distribution penalty", "epsilon"]
    tools.set_metrics(metrics_list)

    report = config['config']['report']
    steps_clip = config['train']['steps_clip']
    total_episodes = int(config['train']['total_episodes'])

    logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    env = CustomEnvironment(reader, config)
    _, sched_obs, none_assignment = env.reset()
    sched_obs_dim = len(sched_obs)
    sched_action_dim = len(sched_obs)
    scheduler = Scheduler(sched_obs_dim, sched_action_dim, config)
    sched_mask = [1 for _ in sched_obs]
    t0 = time.perf_counter()
    for episode in tqdm(range(total_episodes), desc=f"Training"):
        iters = 0
        epsilon = scheduler.set_epsilon(episode)
        t0_ep = time.perf_counter()
        # actions = scheduler.take_action(sched_obs, sched_mask)
        # print(actions)
        # exit(1)
        # env.apply_scheduling(sched_obs, sched_mask, actions)
        while len(none_assignment) > 0:
            iters += 1
            if iters > steps_clip:
                break
            print(f"iters {iters} {len(none_assignment)}/{sched_obs_dim} no assignment", [(cid, env.agent_order_dict[cid]) for cid in none_assignment[:5]])
            actions = scheduler.take_action(sched_obs, sched_mask)
            # if '256' in none_assignment:
            #     ind = env.cid2ind['256']
            #     oi = env.agent_order_dict['256']
            #     print(f"Cid 256->ind {ind}->order {oi}->mask {sched_mask[ind]}")
            _ = env.reset_step(sched_obs, sched_mask, actions)
            # if '256' in none_assignment:
            #     ind = env.cid2ind['256']
            #     oi = env.agent_order_dict['256']
            #     print(f"sched_obs {sched_obs[ind]}->action {actions[ind]}->order {oi}")
            result = env.step()
            _none_assignment = result['not assignment']
            next_states = result['scheduler_observations']
            sched_mask = result['scheduler_mask']
            rewards = result['rewards']
            buffers = {
                'states': sched_obs,
                'actions': actions,
                'next_states': next_states,
                'rewards': rewards,
            }
            scheduler.update(buffers)
            # if '256' in none_assignment:
            #     ind = env.cid2ind['256']
            #     oi = env.agent_order_dict['256']
            #     print(f"action {actions[ind]}->rewards {rewards[ind]}->next_states {next_states[ind]}")
            sched_obs = next_states
            none_assignment = _none_assignment
        runtime = time.perf_counter() - t0_ep
        if len(none_assignment) > 0:
            continue
        metrics = {
            "runtime": runtime,
            "episode_lengths": iters,
            "epsilon": epsilon
        }
        result.update(metrics)
        isbest = update_metrics(result, none_assignment, env, pname, output_folder, runtime)
        if isbest: 
            scheduler.save(pname)
        if isbest and quickrun:
            break
        _, sched_obs, none_assignment = env.reset()

    runtime = time.perf_counter() - t0

    conclude(runtime, report, pname, output_folder)
