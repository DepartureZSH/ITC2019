from dataReader import PSTTReader
from Solution_writter import export_solution_xml
import logging
import pathlib
import os
import yaml
import torch
import time
from math import inf
from validator import report_result
from math import inf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from Solution_writter import export_solution_xml
from validator import report_result
from tools import tools
from MARL.test.env import CustomEnvironment
from MARL.test.network import MAPPO

folder = pathlib.Path(__file__).parent.resolve()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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

def save_to_xml(model, pname, out_path, runtime, config):
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

def CGCS_train(data_folder, output_folder, fileName, trainer, epoches, discount, quickrun=False):
    pname = fileName.split('.xml')[0]
    # 设置log的格式
    setup_logger(pname, f"{output_folder}/{pname}.log")
    logger = logging.getLogger(pname)
    file = f"{data_folder}/{fileName}"
    reader = PSTTReader(file)
    logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    model = trainer(reader, discount)
    # print(f"initial agent order: {model.order_agents()}")
    best_penalty = -inf
    best_result = {}
    t0_ep = time.perf_counter()

    for i in range(epoches + 1):
        result = model.step()
        none_assignment = []
        runtime = time.perf_counter() - t0_ep
        # logger.info(result) # {'not assignment': ['384', '385', '471', '239', '365', '401', '320', '79', '263', '314', '167', '280', '272'], 'penalty': -1032.5950890488828, 'Total_cost': 'TODO', 'Student conflicts': 'TODO', 'Time penalty': 188, 'Room penalty': 239, 'Distribution penalty': 605.595089048882}
        if len(result['not assignment'])==0:
            if result['penalty'] > best_penalty:
                fname = f"{pname}.best_solution.xml"
                out_path = f"{output_folder}/{fname}"
                save_to_xml(model, pname, out_path, runtime, config)
                model.save(f"{output_folder}/{pname}.json")
                logger.info(f"best epoch {i} penalty: {result['penalty']}")
                logger.info(f"best epoch {i} Time penalty: {result['Time penalty']}")
                logger.info(f"best epoch {i} Room penalty: {result['Room penalty']}")
                logger.info(f"best epoch {i} Distribution penalty: {result['Distribution penalty']}")
                best_result = result
            else:
                logger.info(f"valid epoch {i} penalty: {result['penalty']}")
                logger.info(f"valid epoch {i} Time penalty: {result['Time penalty']}")
                logger.info(f"valid epoch {i} Room penalty: {result['Room penalty']}")
                logger.info(f"valid epoch {i} Distribution penalty: {result['Distribution penalty']}")
            if quickrun:
                break
        else:
            # none_assignment = result['not assignment']
            print(f"epoch {i} no assignment values: {model.get_agent_values(result['not assignment'][:10])}")
            # print(f"epoch {i} agent order: {model.order_agents()}")
            logger.info(f"epoch {i} no assignment: {len(result['not assignment'])}")
            logger.info(f"epoch {i} penalty: {result['penalty']}")
            logger.info(f"epoch {i} Time penalty: {result['Time penalty']}")
            logger.info(f"epoch {i} Room penalty: {result['Room penalty']}")
            logger.info(f"epoch {i} Distribution penalty: {result['Distribution penalty']}")
        if i == epoches: 
            print("Training completed.")
            break
        model.reset()
        logger.info("====================================================================")

    runtime = time.perf_counter() - t0_ep
    logger.info("results:")
    logger.info(f"Total runtime: {runtime}")
    if best_penalty > -inf:
        logger.info(f"best epoch penalty: {best_result['penalty']}")
        logger.info(f"best epoch Time penalty: {best_result['Time penalty']}")
        logger.info(f"best epoch Room penalty: {best_result['Room penalty']}")
        logger.info(f"best epoch Distribution penalty: {best_result['Distribution penalty']}")
        validor_result = report_result(f"{output_folder}/{pname}.best_solution.xml")
        logger.info(f"Validation result: {validor_result}")
    else:
        logger.info(f"no valid assignments:")
        fname = f"{pname}.last_solution.xml"
        out_path = f"{output_folder}/{fname}"
        save_to_xml(model, pname, out_path, runtime, config)
        validor_result = report_result(out_path)
        logger.info(f"Validation result: {validor_result}")

def Random_train(data_folder, output_folder, fileName, env, model_config, epoches, discount, quickrun=False):
    pname = fileName.split('.xml')[0]
    # 设置log的格式
    setup_logger(pname, f"{output_folder}/{pname}.log")
    logger = logging.getLogger(pname)
    file = f"{data_folder}/{fileName}"
    reader = PSTTReader(file)
    logger.info(f"{reader.path.name} with {len(reader.courses)} courses, {len(reader.classes)} classes, {len(reader.rooms)} rooms, {len(reader.students)} students, {len(reader.distributions['hard_constraints'])} hard distributions, {len(reader.distributions['soft_constraints'])} soft distributions")
    model = env(reader, discount)
    # print(f"initial agent order: {model.order_agents()}")
    best_penalty = inf
    every_result = []
    best_result = {}
    t0_ep = time.perf_counter()

    for i in range(epoches + 1):
        result = model.step()
        none_assignment = []
        runtime = time.perf_counter() - t0_ep
        # logger.info(result) # {'not assignment': ['384', '385', '471', '239', '365', '401', '320', '79', '263', '314', '167', '280', '272'], 'penalty': -1032.5950890488828, 'Total_cost': 'TODO', 'Student conflicts': 'TODO', 'Time penalty': 188, 'Room penalty': 239, 'Distribution penalty': 605.595089048882}
        if len(result['not assignment'])==0:
            if result['penalty'] < best_penalty:
                fname = f"{pname}.best_solution.xml"
                out_path = f"{output_folder}/{fname}"
                model.save(f"{output_folder}/{pname}.json")
                save_to_xml(model, pname, out_path, runtime, config)
                logger.info(f"best epoch {i} penalty: {result['penalty']}")
                logger.info(f"best epoch {i} Time penalty: {result['Time penalty']}")
                logger.info(f"best epoch {i} Room penalty: {result['Room penalty']}")
                logger.info(f"best epoch {i} Distribution penalty: {result['Distribution penalty']}")
                best_result = result
                best_penalty = result['penalty']
            else:
                logger.info(f"valid epoch {i} penalty: {result['penalty']}")
                logger.info(f"valid epoch {i} Time penalty: {result['Time penalty']}")
                logger.info(f"valid epoch {i} Room penalty: {result['Room penalty']}")
                logger.info(f"valid epoch {i} Distribution penalty: {result['Distribution penalty']}")
            if quickrun:
                break
        else:
            # none_assignment = result['not assignment']
            print(f"epoch {i} no assignment values: {model.get_agent_values(result['not assignment'][:10])}")
            # print(f"epoch {i} agent order: {model.order_agents()}")
            logger.info(f"epoch {i} no assignment: {len(result['not assignment'])}")
            logger.info(f"epoch {i} penalty: {result['penalty']}")
            logger.info(f"epoch {i} Time penalty: {result['Time penalty']}")
            logger.info(f"epoch {i} Room penalty: {result['Room penalty']}")
            logger.info(f"epoch {i} Distribution penalty: {result['Distribution penalty']}")
        every_result.append(result)
        if i == epoches:
            model.save(f"{output_folder}/{pname}.json")
            fname = f"{pname}.last_solution.xml"
            out_path = f"{output_folder}/{fname}"
            save_to_xml(model, pname, out_path, runtime, config)
            print("Training completed.")
            break
        model.reset()
        logger.info("====================================================================")
        # exit(1)

    runtime = time.perf_counter() - t0_ep
    logger.info("results:")
    logger.info(f"Total runtime: {runtime}")
    if best_penalty < inf:
        logger.info(f"best epoch penalty: {best_result['penalty']}")
        logger.info(f"best epoch Time penalty: {best_result['Time penalty']}")
        logger.info(f"best epoch Room penalty: {best_result['Room penalty']}")
        logger.info(f"best epoch Distribution penalty: {best_result['Distribution penalty']}")
        validor_result = report_result(f"{output_folder}/{pname}.best_solution.xml")
        logger.info(f"Validation result: {validor_result}")
    else:
        logger.info(f"no valid assignments!")
        logger.info(f"last epoch penalty: {every_result[-1]['penalty']}")
        logger.info(f"last epoch Time penalty: {every_result[-1]['Time penalty']}")
        logger.info(f"last epoch Room penalty: {every_result[-1]['Room penalty']}")
        logger.info(f"last epoch Distribution penalty: {every_result[-1]['Distribution penalty']}")
        validor_result = report_result(out_path)
        logger.info(f"Validation result: {validor_result}")

def startup(data_folder, output_folder, fileName):
    pname = fileName.split('.xml')[0]
    # 设置log的格式
    os.makedirs(f"{output_folder}/{pname}", exist_ok=True)
    setup_logger(pname, f"{output_folder}/{pname}/{pname}.log")
    logger = logging.getLogger(pname)
    file = f"{data_folder}/{fileName}"
    reader = PSTTReader(file)
    return reader, logger

def main(config):
    if config['method']['name'] == "CGCS":
        output_folder = config['config']['output']
        epoches = config["method"]["epoch"]
        discount = config["method"].get("discount", 1.0)
        quickrun = config["method"].get("quickrun", False)
        from CGCS.train import trainer
        if config['data']['isthrough']:
            data_folder = config["data"]["folder"]
            for fileName in os.listdir(data_folder):
                if fileName.endswith('.xml'):
                    CGCS_train(data_folder, output_folder, fileName, trainer, epoches, discount, quickrun)
        else:
            data_folder = config["data"]["folder"]
            fileName = config["data"]["file"]
            CGCS_train(data_folder, output_folder, fileName, trainer, epoches, discount, quickrun)
    elif config['method']['name'] == "Random":
        output_folder = config['config']['output']
        quickrun = config["method"].get("quickrun", False)
        from MARL.Random.train import train
        if config['data']['isthrough']:
            data_folder = config["data"]["folder"]
            for fileName in os.listdir(data_folder):
                if fileName.endswith('.xml'):
                    reader, logger = startup(data_folder, output_folder, fileName)
                    Tools = tools(logger, config)
                    train(reader, logger, Tools, output_folder, fileName, config, quickrun)
        else:
            data_folder = config["data"]["folder"]
            fileName = config["data"]["file"]
            reader, logger = startup(data_folder, output_folder, fileName)
            Tools = tools(logger, config)
            train(reader, logger, Tools, output_folder, fileName, config, quickrun)
    elif config['method']['name'] == "PLearning":
        output_folder = config['config']['output']
        quickrun = config["method"].get("quickrun", False)
        from MARL.PLearning.train import train
        if config['data']['isthrough']:
            data_folder = config["data"]["folder"]
            for fileName in os.listdir(data_folder):
                if fileName.endswith('.xml'):
                    reader, logger = startup(data_folder, output_folder, fileName)
                    Tools = tools(logger, config)
                    train(reader, logger, Tools, output_folder, fileName, config, quickrun)
        else:
            data_folder = config["data"]["folder"]
            fileName = config["data"]["file"]
            reader, logger = startup(data_folder, output_folder, fileName)
            Tools = tools(logger, config)
            train(reader, logger, Tools, output_folder, fileName, config, quickrun)
    elif config['method']['name'] == "PMAPPO":
        output_folder = config['config']['output']
        quickrun = config["method"].get("quickrun", False)
        from MARL.PMAPPO.train import train
        if config['data']['isthrough']:
            data_folder = config["data"]["folder"]
            for fileName in os.listdir(data_folder):
                if fileName.endswith('.xml'):
                    reader, logger = startup(data_folder, output_folder, fileName)
                    Tools = tools(logger, config)
                    train(reader, logger, Tools, output_folder, fileName, config, quickrun)
        else:
            data_folder = config["data"]["folder"]
            fileName = config["data"]["file"]
            reader, logger = startup(data_folder, output_folder, fileName)
            Tools = tools(logger, config)
            train(reader, logger, Tools, output_folder, fileName, config, quickrun)
    elif config['method']['name'] == "RPMAPPO":
        output_folder = config['config']['output']
        quickrun = config["method"].get("quickrun", False)
        from MARL.RPMAPPO.train import train
        if config['data']['isthrough']:
            data_folder = config["data"]["folder"]
            for fileName in os.listdir(data_folder):
                if fileName.endswith('.xml'):
                    reader, logger = startup(data_folder, output_folder, fileName)
                    Tools = tools(logger, config)
                    train(reader, logger, Tools, output_folder, fileName, config, quickrun)
        else:
            data_folder = config["data"]["folder"]
            fileName = config["data"]["file"]
            reader, logger = startup(data_folder, output_folder, fileName)
            Tools = tools(logger, config)
            train(reader, logger, Tools, output_folder, fileName, config, quickrun)
    elif config['method']['name'] == "P-MAPPO":
        output_folder = config['config']['output']
        epoches = config["method"]["epoch"]
        quickrun = config["method"].get("quickrun", False)
        from MARL.MAPPO.position.train import Position_MAPPO_train
        if config['data']['isthrough']:
            data_folder = config["data"]["folder"]
            for fileName in os.listdir(data_folder):
                if fileName.endswith('.xml'):
                    reader, logger = startup(data_folder, output_folder, fileName)
                    Position_MAPPO_train(reader, logger, fileName, config, epoches, quickrun)
                    # MAPPO_train(data_folder, output_folder, fileName, CustomEnvironment, model_config, epoches, quickrun)
        else:
            data_folder = config["data"]["folder"]
            fileName = config["data"]["file"]
            reader, logger = startup(data_folder, output_folder, fileName)
            Position_MAPPO_train(reader, logger, fileName, config, epoches, quickrun)
    elif config['method']['name'] == "H-MAPPO":
        output_folder = config['config']['output']
        epoches = config["method"]["epoch"]
        quickrun = config["method"].get("quickrun", False)
        from MARL.MAPPO.Hierarchical2.train import Hierarchical_MAPPO_train
        if config['data']['isthrough']:
            data_folder = config["data"]["folder"]
            for fileName in os.listdir(data_folder):
                if fileName.endswith('.xml'):
                    reader, logger = startup(data_folder, output_folder, fileName)
                    Hierarchical_MAPPO_train(reader, logger, fileName, config, epoches, quickrun)
                    # MAPPO_train(data_folder, output_folder, fileName, CustomEnvironment, model_config, epoches, quickrun)
        else:
            data_folder = config["data"]["folder"]
            fileName = config["data"]["file"]
            reader, logger = startup(data_folder, output_folder, fileName)
            Hierarchical_MAPPO_train(reader, logger, fileName, config, epoches, quickrun)
    elif config['method']['name'] == "restart":
        output_folder = config['config']['output']
        epoches = config["method"]["epoch"]
        quickrun = config["method"].get("quickrun", False)
        from MARL.restart.train import MAPPO_train
        if config['data']['isthrough']:
            data_folder = config["data"]["folder"]
            for fileName in os.listdir(data_folder):
                if fileName.endswith('.xml'):
                    reader, logger = startup(data_folder, output_folder, fileName)
                    Tools = tools(logger, config)
                    MAPPO_train(reader, logger, Tools, fileName, config, epoches, quickrun)
        else:
            data_folder = config["data"]["folder"]
            fileName = config["data"]["file"]
            reader, logger = startup(data_folder, output_folder, fileName)
            Tools = tools(logger, config)
            MAPPO_train(reader, logger, Tools, fileName, config, epoches, quickrun)

if __name__ == "__main__":
    # Load configuration
    config = load_cfg(f"{folder}/config.yaml")
    # Set device
    device = torch.device("cuda" if config['device'] == "gpu" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main(config)