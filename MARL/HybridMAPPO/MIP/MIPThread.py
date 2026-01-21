import threading
import logging
from MARL.HybridMAPPO.MIP.solver import MIPSolver

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

class MIPThread(threading.Thread):
    def __init__(self, solution_queue, output_folder, pname, reader, config):
        super().__init__()
        self.solver_id = "MIP"
        self.solution_queue = solution_queue
        self.output_folder = output_folder
        self.pname = pname
        setup_logger(self.solver_id, f"{output_folder}/logger.log")
        self.logger = logging.getLogger(self.solver_id)
        self.reader = reader
        self.config = config
        self.daemon = True
        
    def run(self):
        self.logger.info(f"Solver-{self.solver_id} start")

        solver = MIPSolver(self.reader, self.logger, self.config)

        # 构建模型
        if self.config['train']['MIP'].get('reproduction', False):
            model_path = f'{self.output_folder}/model/{self.pname}'
            solver.load_model(model_path)
        else:
            solver.build_model()

        # 求解
        epoch = int(self.config['train']['MIP']['epoch'])
        Solutions = int(self.config['train']['MIP']['Solutions'])
        c = 1
        for i in range(epoch):
            assignments_list = solver.solve()

            if len(assignments_list) > 0:
                output_path = f'{self.output_folder}/model/{self.pname}'
                solver.save_model(output_path)

            for assignments in assignments_list:
                output_file = f'{self.output_folder}/solution{c}_{self.pname}.xml'
                if c >= Solutions:
                    self.logger.info(f"===={Solutions} solution reach====")
                    return
                c += 1
                solver.save_solution(assignments, output_file, self.config)
                self.logger.info(f"Solver-{self.solver_id} generate a valid soltion: {output_file}. Add to Optimize...")
                self.solution_queue.put(output_file)