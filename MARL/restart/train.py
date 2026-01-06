from math import inf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from validator import report_result
from MARL.restart.env import CustomEnvironment
from MARL.restart.network import MAPPO, Scheduler

def MAPPO_train(reader, logger, tools, fileName, config, epoches, quickrun):
    pname = fileName.split('.xml')[0]
    # obs_dim = config['train']['obs_dim']
    save_to_xml = tools.save_to_xml
    plot_all_metrics = tools.plot_all_metrics 
    action_dim = config['train']['action_dim']
    sched_action_dim = config['train']['sched_action_dim']

    env = CustomEnvironment(reader, action_dim)
    # 初始化环境
    observations, not_assignment = env.reset()
    agents_order = env.agents_order
    
    team_size = len(agents_order.keys())

    obs_dim = len(agents_order.keys())
    

    # 初始化MAPPO
    env_name = config['train']['env_name']
    actor_lr = float(config['train']['actor_lr'])
    critic_lr = float(config['train']['critic_lr'])
    hidden_dim = int(config['train']['hidden_dim'])
    gamma = float(config['train']['gamma'])
    lmbda = float(config['train']['lmbda'])
    eps = float(config['train']['eps'])
    device = config['train']['device']
    output_dir = config['config']['output']
    output_dir = f"{output_dir}/{pname}"

    os.makedirs(output_dir, exist_ok=True)
    total_episodes = int(config['train']['total_episodes'])
    episodes_avg_num = int(config['train']['episodes_statistics'])
    steps_clip = int(config['train']['steps_clip'])

    scheduler = Scheduler(obs_dim, hidden_dim, sched_action_dim,
                 actor_lr, critic_lr, lmbda, eps, gamma, device, output_dir)
    mappo = MAPPO(team_size, obs_dim, hidden_dim, action_dim,
                  actor_lr, critic_lr, lmbda, eps, gamma, device, output_dir)
    
    best_penalty = inf
    best_result_i = -1

    # 用于统计指标的列表
    total_rewards_per_episode = []
    total_rewards_sched_per_episode = []
    total_penalty_per_episode = []
    total_Time_penalty_per_episode = []
    total_Room_penalty_per_episode = []
    total_Distribution_penalty_per_episode = []
    
    episode_lengths = []
    policy_losses = []
    value_losses = []
    entropies = []
    policy_losses_sched = []
    value_losses_sched = []
    entropies_sched = []

    # 每{episodes_avg_num}个episode的平均值列表
    avg_total_rewards_per = []
    avg_rewards_sched_per = []
    avg_episode_length_per = []
    avg_policy_loss_per = []
    avg_value_loss_per = []
    avg_entropy_per = []
    avg_policy_loss_sched_per = []
    avg_value_loss_sched_per = []
    avg_entropy_sched_per = []
    avg_penalty_per = []
    t0 = time.perf_counter()

    for episode in tqdm(range(total_episodes), desc=f"Training none assignment {len(not_assignment)}"):
        # 初始化Trajectory buffer
        buffers = [{
            'states': [], 
            'actions': [], 
            'next_states': [], 
            'rewards': [], 
            'dones': [], 
            'action_probs': [],
        } for _ in range(team_size)]
        sched_buffer = {
            'states': [], 
            'actions': [], 
            'next_states': [], 
            'rewards': [], 
            'dones': [], 
            'action_probs': []
        }

        obs, not_assignment = env.reset()
        agent_order = [cid for cid, _ in env.agents_order.values()]

        ep_reward = 0
        ep_penalty = 0
        ep_Time_penalty = 0
        ep_Room_penalty = 0
        ep_Distribution_penalty = 0
        iters = 0
        t0_ep = time.perf_counter()

        # env.order_agents()
        # agent_order = [cid for cid, _ in env.agents_order.values()]
        # env.step_reset()
        # infos = env.step()

        # not_assignment_ = not_assignment

        # not_assignment = infos['not assignment']
        # rewards = infos['rewards']
        # total_rewards = infos['total_rewards']
        # penalty = infos['penalty']
        # Time_penalty = infos['Time penalty']
        # Room_penalty = infos['Room penalty']
        # Distribution_penalty = infos['Distribution penalty']
        # next_obs = infos['observations']

        # ep_reward = total_rewards
        # ep_penalty = penalty
        # ep_Time_penalty = Time_penalty
        # ep_Room_penalty = Room_penalty
        # ep_Distribution_penalty = Distribution_penalty
        
        # obs = next_obs
        # print(f"iters: 0 has {len(not_assignment)}/{len(agent_order)} no assignment")

        while len(not_assignment) > 0:
            iters += 1
            if iters >= steps_clip:
                break
            sched_state_list = [obs[cid].flatten() for cid in not_assignment]
            sched_actions, sched_probs = scheduler.take_action(sched_state_list)
            sched_action_dict = {cid: sched_actions[i] for i, cid in enumerate(not_assignment)}
            env.apply_sched_action(sched_action_dict)
            state_list = [obs[cid].flatten() for cid in agent_order]
            actions, probs = mappo.take_action(state_list)
            action_dict = {cid: actions[i] for i, cid in enumerate(agent_order)}
            print(f"iters: {iters-1} has {len(not_assignment)}/{len(agent_order)} no assignment:", [(cid, action_dict[cid], env.get_agent_value(cid), env.agent_order_dict[cid]) for i, cid in enumerate(not_assignment[:5])])
            env.apply_action(action_dict)
            env.order_agents()
            agent_order = [cid for cid, _ in env.agents_order.values()]
            env.step_reset()
            infos = env.step()

            not_assignment_ = not_assignment

            not_assignment = infos['not assignment']
            rewards = infos['rewards']
            total_rewards = infos['total_rewards']
            penalty = infos['penalty']
            Time_penalty = infos['Time penalty']
            Room_penalty = infos['Room penalty']
            Distribution_penalty = infos['Distribution penalty']
            next_obs = infos['observations']

            ep_reward = total_rewards
            ep_reward_sched = sum([float((len(not_assignment_)-len(not_assignment))) if agent not in not_assignment else 0 for agent in not_assignment_])
            ep_penalty = penalty
            ep_Time_penalty = Time_penalty
            ep_Room_penalty = Room_penalty
            ep_Distribution_penalty = Distribution_penalty
            
            # 存储transition
            for i, agent in enumerate(agent_order):
                buffers[i]['states'].append(np.array(obs[agent]))
                buffers[i]['actions'].append(actions[i])
                buffers[i]['next_states'].append(np.array(next_obs[agent]))
                buffers[i]['rewards'].append(float(rewards[agent]))
                buffers[i]['dones'].append(float(agent not in not_assignment))
                buffers[i]['action_probs'].append(probs[i])
            for i, agent in enumerate(not_assignment_):
                sched_buffer['states'].append(np.array(obs[agent]))
                sched_buffer['actions'].append(sched_actions[i])
                sched_buffer['next_states'].append(np.array(next_obs[agent]))
                sched_buffer['rewards'].append(float((len(not_assignment_)-len(not_assignment))) if agent not in not_assignment else 0)
                sched_buffer['dones'].append(float(agent not in not_assignment))
                sched_buffer['action_probs'].append(sched_probs[i])
            obs = next_obs

            # agents_order = env.agents_order
            # print(agents_order[:5])
            # if iters == 5: exit(1)
        runtime = time.perf_counter() - t0_ep
        # 使用MAPPO更新参数
        sched_a_loss, sched_c_loss, sched_ent = scheduler.update(sched_buffer)
        a_loss, c_loss, ent = mappo.update(buffers, obs_dim)

        # 记录指标
        total_rewards_per_episode.append(ep_reward)
        total_rewards_sched_per_episode.append(ep_reward_sched)
        total_penalty_per_episode.append(ep_penalty)
        total_Time_penalty_per_episode.append(ep_Time_penalty)
        total_Room_penalty_per_episode.append(ep_Room_penalty)
        total_Distribution_penalty_per_episode.append(ep_Distribution_penalty)
        episode_lengths.append(iters)
        policy_losses.append(a_loss)
        value_losses.append(c_loss)
        entropies.append(ent)
        policy_losses_sched.append(sched_a_loss)
        value_losses_sched.append(sched_c_loss)
        entropies_sched.append(sched_ent)

        if len(not_assignment) == 0:
            if ep_penalty < best_penalty:
                fname = f"{pname}.best_solution.xml"
                out_path = f"{output_dir}/{fname}"
                env.save(f"{output_dir}/{pname}.best.json")
                save_to_xml(env, pname, out_path, runtime, config)
                logger.info(f"best Episode {episode+1} penalty: {ep_penalty}")
                logger.info(f"best Episode {episode+1} Time penalty: {ep_Time_penalty}")
                logger.info(f"best Episode {episode+1} Room penalty: {ep_Room_penalty}")
                logger.info(f"best Episode {episode+1} Distribution penalty: {ep_Distribution_penalty}")
                # best_result = result
                best_penalty = ep_penalty
                # 保存模型的权重参数
                model_path = f"{output_dir}/model"
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                mappo.save_model(path=model_path)
                logger.info(f"Model saved at episode {episode}")
            else:
                logger.info(f"valid Episode {episode+1} penalty: {ep_penalty}")
                logger.info(f"valid Episode {episode+1} Time penalty: {ep_Time_penalty}")
                logger.info(f"valid Episode {episode+1} Room penalty: {ep_Room_penalty}")
                logger.info(f"valid Episode {episode+1} Distribution penalty: {ep_Distribution_penalty}")
            if quickrun:
                break
        else:
            # none_assignment = result['not assignment']
            # print(f"Episode {episode+1} no assignment values: {env.get_agent_values(not_assignment[:10])}")
            # print(f"Episode {episode+1} agent order: {model.order_agents()}")
            logger.info(f"Episode {episode+1} no assignment: {len(not_assignment)}")
            logger.info(f"Episode {episode+1} penalty: {ep_penalty}")
            logger.info(f"Episode {episode+1} Time penalty: {ep_Time_penalty}")
            logger.info(f"Episode {episode+1} Room penalty: {ep_Room_penalty}")
            logger.info(f"Episode {episode+1} Distribution penalty: {ep_Distribution_penalty}")
        if episode + 1 == total_episodes:
            fname = f"{pname}.last_solution.xml"
            out_path = f"{output_dir}/{fname}"
            env.save(f"{output_dir}/{pname}.last.json")
            save_to_xml(env, pname, out_path, runtime, config)
        # 每{episodes_avg_num}个episode统计一次平均值并记录日志、绘图
        if episode % episodes_avg_num == 0:
            avg_reward = np.mean(total_rewards_per_episode[-episodes_avg_num:])
            avg_rewards_sched = np.mean(total_rewards_sched_per_episode[-episodes_avg_num:])
            avg_penalty = np.mean(total_penalty_per_episode[-episodes_avg_num:])
            avg_length = np.mean(episode_lengths[-episodes_avg_num:])
            avg_policy_loss = np.mean(policy_losses[-episodes_avg_num:])
            avg_value_loss = np.mean(value_losses[-episodes_avg_num:])
            avg_entropy = np.mean(entropies[-episodes_avg_num:])
            avg_policy_loss_sched = np.mean(policy_losses_sched[-episodes_avg_num:])
            avg_value_loss_sched = np.mean(value_losses_sched[-episodes_avg_num:])
            avg_entropy_sched = np.mean(entropies_sched[-episodes_avg_num:])

            avg_total_rewards_per.append(avg_reward)
            avg_rewards_sched_per.append(avg_rewards_sched)
            avg_episode_length_per.append(avg_length)
            avg_policy_loss_per.append(avg_policy_loss)
            avg_value_loss_per.append(avg_value_loss)
            avg_entropy_per.append(avg_entropy)
            avg_penalty_per.append(avg_penalty)
            avg_policy_loss_sched_per.append(avg_policy_loss_sched)
            avg_value_loss_sched_per.append(avg_value_loss_sched)
            avg_entropy_sched_per.append(avg_entropy_sched)

            logger.info(f"Episode {episode + 1}: "
                        f"AvgEpisodeLength(last{episodes_avg_num})={avg_length:.3f}, "
                        f"AvgPenalty(last{episodes_avg_num})={avg_penalty:.3f}")
            
            logger.info(f"AvgTotalReward(last{episodes_avg_num})={avg_reward:.3f}, "
                        f"AvgPolicyLoss(last{episodes_avg_num})={avg_policy_loss:.3f}, "
                        f"AvgValueLoss(last{episodes_avg_num})={avg_value_loss:.3f}, "
                        f"AvgEntropy(last{episodes_avg_num})={avg_entropy:.3f}")
            
            logger.info(f"AvgTotalReward_sched(last{episodes_avg_num})={avg_rewards_sched:.3f}, "
                        f"AvgPolicyLoss_sched(last{episodes_avg_num})={avg_policy_loss_sched:.3f}, "
                        f"AvgValueLoss_sched(last{episodes_avg_num})={avg_value_loss_sched:.3f}, "
                        f"AvgEntropy_sched(last{episodes_avg_num})={avg_entropy_sched:.3f}")
            
            # 创建指标字典
            metrics_dict = {
                "Average_Total_Reward": avg_total_rewards_per,
                "Average_Episode_Length": avg_episode_length_per,
                "Average_Policy_Loss": avg_policy_loss_per,
                "Average_Value_Loss": avg_value_loss_per, 
                "Average_Entropy": avg_entropy_per,
                "Average_Penalty": avg_penalty_per
            }

            # 创建指标字典
            metrics_dict1 = {
                "Average_Total_Reward_sched": avg_total_rewards_per,
                "Average_Episode_Length_sched": avg_episode_length_per,
                "Average_Policy_Loss_sched": avg_policy_loss_sched_per,
                "Average_Value_Loss_sched": avg_value_loss_sched_per, 
                "Average_Entropy_sched": avg_entropy_sched_per,
                "Average_Penalty": avg_penalty_per
            }

            # 调用新的绘图函数
            plot_all_metrics(env_name, output_dir, "mappo_metrics", metrics_dict, episode, episodes_avg_num)
            plot_all_metrics(env_name, output_dir, "scheduler_metrics", metrics_dict1, episode, episodes_avg_num)
        
        # print(f"Episode {episode+1}: penalty = {ep_penalty}")
        logger.info("====================================================================")

    runtime = time.perf_counter() - t0
    logger.info("results:")
    logger.info(f"Total runtime: {runtime}")
    if best_penalty < inf:
        logger.info(f"best Episode penalty: {total_penalty_per_episode[best_result_i]}")
        logger.info(f"best Episode Time penalty: {total_Time_penalty_per_episode[best_result_i]}")
        logger.info(f"best Episode Room penalty: {total_Room_penalty_per_episode[best_result_i]}")
        logger.info(f"best Episode Distribution penalty: {total_Distribution_penalty_per_episode[best_result_i]}")
        # validor_result = report_result(f"{output_dir}/{pname}.best_solution.xml")
        # logger.info(f"Validation result: {validor_result}")
    else:
        logger.info(f"no valid assignments!")
        logger.info(f"last Episode penalty: {total_penalty_per_episode[-1]}")
        print(total_penalty_per_episode)
        logger.info(f"last Episode Time penalty: {total_Time_penalty_per_episode[-1]}")
        logger.info(f"last Episode Room penalty: {total_Room_penalty_per_episode[-1]}")
        logger.info(f"last Episode Distribution penalty: {total_Distribution_penalty_per_episode[-1]}")
        # validor_result = report_result(f"{output_dir}/{pname}.last_solution.xml")
        # logger.info(f"Validation result: {validor_result}")
    