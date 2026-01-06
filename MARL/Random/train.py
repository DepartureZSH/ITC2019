import time
from tqdm import tqdm
from MARL.Random.env import CustomEnvironment

def train(reader, logger, tools, output_folder, fileName, config, quickrun=False):
    pname = fileName.split('.xml')[0]
    update_metrics = tools.update_metrics
    conclude = tools.conclude
    metrics_list = ["episode_lengths", "runtime", "Total cost", "Time penalty", "Room penalty", "Distribution penalty"]
    tools.set_metrics(metrics_list)

    report = config['config']['report']
    discount = config['train']['discount']
    steps_clip = config['train']['steps_clip']
    total_episodes = int(config['train']['total_episodes'])

    logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    env = CustomEnvironment(reader, discount)
    
    epoch = 1
    t0 = time.perf_counter()

    for episode in tqdm(range(total_episodes), desc=f"Training {epoch}"):
        epoch += 1
        observations, none_assignment = env.reset()
        iters = 0
        result = {}
        t0_ep = time.perf_counter()
        while len(none_assignment) > 0:
            iters += 1
            if iters > steps_clip:
                break
            result = env.step()
            none_assignment = result['not assignment']
            observations = env.reset_step()
        runtime = time.perf_counter() - t0_ep
        if len(none_assignment) > 0:
            continue
        metrics = {
            "runtime": runtime,
            "episode_lengths": iters
        }
        result.update(metrics)
        isbest = update_metrics(result, none_assignment, env, pname, output_folder, runtime)
        if isbest and quickrun:
            break

    runtime = time.perf_counter() - t0

    conclude(runtime, report, pname, output_folder)
