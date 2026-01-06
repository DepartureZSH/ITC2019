from math import inf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from Solution_writter import export_solution_xml
from validator import report_result
from MARL.MAPPO.Hierarchical2.env import CustomEnvironment
from MARL.MAPPO.Hierarchical2.network import MAPPO


def save_to_xml(model, pname, out_path, runtime, config):
    print("save_to_xml", model.total_penalty()["penalty"])
    assignments = model.results()
    export_solution_xml(
        assignments=assignments,   # class_id -> (time_idx, room_id or None)
        out_path=str(out_path),
        name=pname,
        runtime_sec=runtime,
        cores=os.cpu_count(),
        technique=config['config']['technique'],
        author=config['config']['author'],
        institution=config['config']['institution'],
        country=config['config']['country'],
        include_students=config['config']['include_students'],
    )

def plot_all_metrics(env_name, plots_dir, net_name, metrics_dict, episode, episodes_avg_num):
    """
    将所有指标绘制到一个包含多个子图的图表中
    - 对曲线进行平滑处理
    - 添加误差带显示
    参数:
    metrics_dict: 包含所有指标数据的字典，格式为 {metric_name: values_list}
    episode: 当前的episode数
    """
    # 创建一个2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics of {env_name} (Up to Episode {episode})', fontsize=16)
    
    # 压平axes数组以便迭代
    axes = axes.flatten()
    
    # 为每个指标获取x轴值
    any_metric = list(metrics_dict.values())[0]
    x_values = [episodes_avg_num * (i + 1) for i in range(len(any_metric))]
    
    # 平滑参数 - 窗口大小
    window_size = min(6, len(x_values)) if len(x_values) > 0 else 1
    
    # 在每个子图中绘制一个指标
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if i >= 6:  # 我们只有5个指标
            break
            
        ax = axes[i]
        values_array = np.array(values)
        
        # 应用平滑处理
        if len(values) > window_size:
            # 创建平滑曲线
            smoothed = np.convolve(values_array, np.ones(window_size)/window_size, mode='valid')
            
            # 计算滚动标准差用于误差带
            std_values = []
            for j in range(len(values) - window_size + 1):
                std_values.append(np.std(values_array[j:j+window_size]))
            std_values = np.array(std_values)
            
            # 调整x轴以匹配平滑后的数据长度
            smoothed_x = x_values[window_size-1:]
            
            # 绘制平滑曲线和原始散点
            ax.plot(smoothed_x, smoothed, '-', linewidth=2, label='Smoothed')
            ax.scatter(x_values, values, alpha=0.3, label='Original')
            
            # 添加误差带
            ax.fill_between(smoothed_x, smoothed-std_values, smoothed+std_values, 
                           alpha=0.2, label='±1 StdDev')
        else:
            # 如果数据点太少，只绘制原始数据
            ax.plot(x_values, values, 'o-', label='Data')
        
        ax.set_title(metric_name.replace('_', ' '))
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric_name.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 删除未使用的子图
    if len(metrics_dict) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, net_name))
    plt.close(fig)


def Hierarchical_MAPPO_train(reader, logger, fileName, config, epoches, quickrun=False):
    pname = fileName.split('.xml')[0]
    
    obs_shape = config['train']['obs_shape']
    action_dim = config['train']['action_dim']
    # logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    
    env = CustomEnvironment(reader, obs_shape, action_dim)
    # 初始化环境
    observations, done = env.reset()
    agent_order = env._agents
    
    
    team_size = len(agent_order)

    # print("obs_shape, action_dim, team_size", obs_shape, action_dim, team_size)

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
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    total_episodes = int(config['train']['total_episodes'])
    episodes_avg_num = int(config['train']['episodes_statistics'])
    steps_clip = int(config['train']['steps_clip'])
    
    mappo = MAPPO(team_size, obs_shape, hidden_dim, action_dim,
                  actor_lr, critic_lr, lmbda, eps, gamma, device, output_dir)
    
    best_penalty = inf
    best_result_i = -1

    # 用于统计指标的列表
    total_rewards_per_episode = []
    total_penalty_per_episode = []
    total_Time_penalty_per_episode = []
    total_Room_penalty_per_episode = []
    total_Distribution_penalty_per_episode = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    entropies = []

    # 每{episodes_avg_num}个episode的平均值列表
    avg_total_rewards_per = []
    avg_episode_length_per = []
    avg_policy_loss_per = []
    avg_value_loss_per = []
    avg_entropy_per = []
    avg_penalty_per = []
    t0 = time.perf_counter()

    for episode in tqdm(range(total_episodes), desc=f"Training none assignment {len(done)}"):
        # 初始化Trajectory buffer
        buffers = [{
            'states': [], 
            'actions': [], 
            'next_states': [], 
            'rewards': [], 
            'dones': [], 
            'action_probs': [],
        } for _ in range(team_size)]

        obs, done = env.reset()
        ep_reward = 0
        ep_penalty = 0
        ep_Time_penalty = 0
        ep_Room_penalty = 0
        ep_Distribution_penalty = 0
        iters = 0

        t0_ep = time.perf_counter()
        while len(done) > 0:
            iters += 1
            if iters % steps_clip == 0:
                break
            state_list = [obs[cid].flatten() for cid in agent_order]
            actions, probs = mappo.take_action(state_list)
            action_dict = {cid: actions[i] for i, cid in enumerate(agent_order)}
            env.move_agent(action_dict)
            if iters % 50 == 1:
                print("agents_order:", env.agents_order[:5])
            infos = env.step()
            done = infos['not assignment']
            rewards = infos['rewards']
            penalty = infos['penalty']
            penalty_agent = infos['penalty_agent']
            Time_penalty = infos['Time penalty']
            Room_penalty = infos['Room penalty']
            Distribution_penalty = infos['Distribution penalty']
            next_obs = infos['observations']
            ep_reward = -penalty
            ep_penalty = penalty
            ep_Time_penalty = Time_penalty
            ep_Room_penalty = Room_penalty
            ep_Distribution_penalty = Distribution_penalty
            
            # 存储transition
            for i, agent in enumerate(agent_order):
                buffers[i]['states'].append(np.array(obs[agent]))
                buffers[i]['actions'].append(actions[i])
                buffers[i]['next_states'].append(np.array(next_obs[agent]))
                buffers[i]['rewards'].append(float(penalty_agent[agent]))
                buffers[i]['dones'].append(float(agent in done))
                buffers[i]['action_probs'].append(probs[i])
            obs = next_obs

            if iters % 10 == 1:
                print(f"iters: {iters-1} has {len(done)}/{len(agent_order)} no assignment:", [(cid, actions[i], ) for i, cid in enumerate(done[:5])])

        runtime = time.perf_counter() - t0_ep
        # 使用MAPPO更新参数
        a_loss, c_loss, ent = mappo.update(buffers, obs_shape)

        # 记录指标
        total_rewards_per_episode.append(ep_reward)
        total_penalty_per_episode.append(ep_penalty)
        total_Time_penalty_per_episode.append(ep_Time_penalty)
        total_Room_penalty_per_episode.append(ep_Room_penalty)
        total_Distribution_penalty_per_episode.append(ep_Distribution_penalty)
        episode_lengths.append(iters)
        policy_losses.append(a_loss)
        value_losses.append(c_loss)
        entropies.append(ent)

        if len(done) == 0:
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
            print(f"Episode {episode+1} no assignment values: {env.get_agent_values(done[:10])}")
            # print(f"Episode {episode+1} agent order: {model.order_agents()}")
            logger.info(f"Episode {episode+1} no assignment: {len(done)}")
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
            avg_penalty = np.mean(total_penalty_per_episode[-episodes_avg_num:])
            avg_length = np.mean(episode_lengths[-episodes_avg_num:])
            avg_policy_loss = np.mean(policy_losses[-episodes_avg_num:])
            avg_value_loss = np.mean(value_losses[-episodes_avg_num:])
            avg_entropy = np.mean(entropies[-episodes_avg_num:])

            avg_total_rewards_per.append(avg_reward)
            avg_episode_length_per.append(avg_length)
            avg_policy_loss_per.append(avg_policy_loss)
            avg_value_loss_per.append(avg_value_loss)
            avg_entropy_per.append(avg_entropy)
            avg_penalty_per.append(avg_penalty)

            logger.info(f"Episode {episode + 1}: "
                        f"AvgEpisodeLength(last{episodes_avg_num})={avg_length:.3f}, "
                        f"AvgPenalty(last{episodes_avg_num})={avg_penalty:.3f}")
            
            logger.info(f"AvgTotalReward(last{episodes_avg_num})={avg_reward:.3f}, "
                        f"AvgPolicyLoss(last{episodes_avg_num})={avg_policy_loss:.3f}, "
                        f"AvgValueLoss(last{episodes_avg_num})={avg_value_loss:.3f}, "
                        f"AvgEntropy(last{episodes_avg_num})={avg_entropy:.3f}")
            
            # 创建指标字典
            metrics_dict = {
                "Average_Total_Reward": avg_total_rewards_per,
                "Average_Episode_Length": avg_episode_length_per,
                "Average_Policy_Loss": avg_policy_loss_per,
                "Average_Value_Loss": avg_value_loss_per, 
                "Average_Entropy": avg_entropy_per,
                "Average_Penalty": avg_penalty_per
            }

            # 调用新的绘图函数
            plot_all_metrics(env_name, output_dir, "mappo_metrics", metrics_dict, episode, episodes_avg_num)
        
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
        logger.info(f"last Episode Time penalty: {total_Time_penalty_per_episode[-1]}")
        logger.info(f"last Episode Room penalty: {total_Room_penalty_per_episode[-1]}")
        logger.info(f"last Episode Distribution penalty: {total_Distribution_penalty_per_episode[-1]}")
        # validor_result = report_result(f"{output_dir}/{pname}.last_solution.xml")
        # logger.info(f"Validation result: {validor_result}")

if __name__ == "__main__":
    from dataReader import PSTTReader
    file = "/home/scxsz1/zsh/itc2019/data/late/muni-fi-fal17.xml"
    reader = PSTTReader(file)
    train = Position_MAPPO_train(reader, fileName="muni-fi-fal17.xml", output_folder="./", epoches=10)
