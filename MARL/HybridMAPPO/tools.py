import os
import json
import numpy as np
from math import inf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
from Solution_writter import export_solution_xml
from validator import report_result

class tools:
    def __init__(self, logger=None, config=None):
        self.logger = logger
        self.config = config
        self.best_cost = inf
        self.best_result = {}
        self.last_result = {}
        self.metrics = {}

    def set_metrics(self, metrics_list):
        for key in metrics_list:
            self.metrics[key] = []

    def save_to_xml(self, model, pname, out_path, runtime, config):
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

    def assignments_to_xml(self, assignments, pname, out_path, runtime):
        export_solution_xml(
            assignments=assignments,   # class_id -> (time_idx, room_id or None)
            out_path=str(out_path),
            name=pname,
            runtime_sec=runtime,
            cores=os.cpu_count(),
            technique=self.config['config']['technique'],
            author=self.config['config']['author'],
            institution=self.config['config']['institution'],
            country=self.config['config']['country'],
            include_students=self.config['config']['include_students'],
        )

    def plot_metrics(self, output_folder):
        env_name = self.config['train']['env_name']
        net_name = f"{env_name}_metrics"
        plots_dir = output_folder
        layout_dict = {
            1: [1, 1],
            2: [1, 2],
            3: [1, 3],
            4: [2, 2],
            5: [2, 3],
            6: [2, 3],
            7: [3, 3],
            8: [3, 3],
            9: [3, 3],
            10: [4, 3],
            11: [4, 3],
            12: [4, 3]
        }
        metrics_dict = self.metrics
        if len(metrics_dict) > len(layout_dict):
            print("Metrics_dict length overflow! Please assign a proper layout first.")
        layout = layout_dict[len(metrics_dict)]
        # 创建子图布局
        fig, axes = plt.subplots(layout[0], layout[1], figsize=(18, 10))
        fig.suptitle(f'Training Metrics of {env_name}', fontsize=16)
        
        # 压平axes数组以便迭代
        axes = axes.flatten()
        
        # 为每个指标获取x轴值
        any_metric = list(metrics_dict.values())[0]
        x_values = [i + 1 for i in range(len(any_metric))]
        
        # 在每个子图中绘制一个指标
        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            ax = axes[i]
            ax.plot(x_values, values, 'o-', label=metric_name)
            
            ax.set_title(metric_name.replace('_', ' '))
            ax.set_xlabel('Episodes')
            ax.set_ylabel(metric_name.replace('_', ' '))
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 删除未使用的子图
        for i in range(len(metrics_dict), layout[0] * layout[1]):
            fig.delaxes(axes[i])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(plots_dir, net_name))
        plt.close(fig)
        with open(os.path.join(plots_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def update_metrics(self, result, none_assignment_num, env, pname, output_folder, runtime):
        for key in self.metrics.keys():
            self.metrics[key].append(result[key])
        self.plot_metrics(output_folder)

        self.last_result = result

        if none_assignment_num==0:
            if result['Best Total Cost'] < self.best_cost:
                assignments = env.results()
                env.save(f"{output_folder}/{pname}.json")
                out_path = f"{output_folder}/{pname}.best_solution.xml"
                self.assignments_to_xml(assignments, pname, out_path, runtime)
                self.logger.info(f"best Episode Total cost: {result['Best Total Cost']}")
                self.logger.info(f"best Episode Time penalty: {result['Best Time Penalty']}")
                self.logger.info(f"best Episode Room penalty: {result['Best Room Penalty']}")
                self.logger.info(f"best Episode Distribution penalty: {result['Best Distribution Penalty']}")
                self.logger.info("====================================================================")
                self.best_result = result
                self.best_cost = result['Best Total Cost']
                return True
            else:
                assignments = env.results()
                env.save(f"{output_folder}/{pname}.json")
                out_path = f"{output_folder}/{pname}.last_solution.xml"
                self.logger.info(f"valid Episode Total cost: {result['Best Total Cost']}")
                self.logger.info(f"valid Episode Time penalty: {result['Best Time Penalty']}")
                self.logger.info(f"valid Episode Room penalty: {result['Best Room Penalty']}")
                self.logger.info(f"valid Episode Distribution penalty: {result['Best Distribution Penalty']}")
                self.logger.info("====================================================================")
                return False
        else:
            assignments = env.results()
            env.save(f"{output_folder}/{pname}.json")
            out_path = f"{output_folder}/{pname}.last_solution.xml"
            self.logger.info(f"Episode Total cost: {result['Best Total Cost']}")
            self.logger.info(f"Episode no assignment: {none_assignment_num}")
            self.logger.info(f"Episode Time penalty: {result['Best Time Penalty']}")
            self.logger.info(f"Episode Room penalty: {result['Best Room Penalty']}")
            self.logger.info(f"Episode Distribution penalty: {result['Best Distribution Penalty']}")
            self.logger.info("====================================================================")
            return False
    
    def conclude(self, runtime, report, pname, output_folder):
        self.logger.info("results:")
        self.logger.info(f"Total runtime: {runtime}")
        if self.best_cost < inf:
            self.logger.info(f"best Episode Total cost: {self.best_result['Best Total Cost']}")
            self.logger.info(f"best Episode Time penalty: {self.best_result['Best Time Penalty']}")
            self.logger.info(f"best Episode Room penalty: {self.best_result['Best Room Penalty']}")
            self.logger.info(f"best Episode Distribution penalty: {self.best_result['Best Distribution Penalty']}")
            if report:
                validor_result = report_result(f"{output_folder}/{pname}.best_solution.xml")
                self.logger.info(f"Validation result: {validor_result}")
        else:
            self.logger.info(f"no valid assignments!")
            self.logger.info(f"best Episode Total cost: {self.last_result['Best Total Cost']}")
            self.logger.info(f"last Episode Time penalty: {self.last_result['Best Time Penalty']}")
            self.logger.info(f"last Episode Room penalty: {self.last_result['Best Room Penalty']}")
            self.logger.info(f"last Episode Distribution penalty: {self.last_result['Best Distribution Penalty']}")
            if report:
                validor_result = report_result(f"{output_folder}/{pname}.last_solution.xml")
                self.logger.info(f"Validation result: {validor_result}")