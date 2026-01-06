import os
import json
import numpy as np
from math import inf
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

    def plot_all_metrics(self, env_name, plots_dir, net_name, metrics_dict, episode, episodes_avg_num):
        """
        将所有指标绘制到一个包含多个子图的图表中
        - 对曲线进行平滑处理
        - 添加误差带显示
        参数:
        metrics_dict: 包含所有指标数据的字典，格式为 {metric_name: values_list}
        episode: 当前的episode数
        """
        layout_dict = {
            1: [1, 1],
            2: [1, 2],
            3: [1, 3],
            4: [2, 2],
            5: [2, 3],
            6: [2, 3],
            7: [3, 3],
            8: [3, 3],
            9: [3, 3]
        }
        if len(metrics_dict) > 9:
            print("Metrics_dict length overflow! Please assign a proper layout first.")
        layout = layout_dict[len(metrics_dict)]
        # 创建子图布局
        fig, axes = plt.subplots(layout[0], layout[1], figsize=(18, 10))
        fig.suptitle(f'Training Metrics of {env_name} (Up to Episode {episode})', fontsize=16)
        
        # 压平axes数组以便迭代
        axes = axes.flatten()
        
        # 为每个指标获取x轴值
        any_metric = list(metrics_dict.values())[0]
        x_values = [episodes_avg_num * (i + 1) for i in range(len(any_metric))]
        
        # 平滑参数 - 窗口大小
        window_size = 1000
        
        # 在每个子图中绘制一个指标
        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            ax = axes[i]
            # values_array = np.array(values)
            
            # # 应用平滑处理
            # if len(values) > window_size:
            #     # 创建平滑曲线
            #     smoothed = np.convolve(values_array, np.ones(window_size)/window_size, mode='valid')
                
            #     # 计算滚动标准差用于误差带
            #     std_values = []
            #     for j in range(len(values) - window_size + 1):
            #         std_values.append(np.std(values_array[j:j+window_size]))
            #     std_values = np.array(std_values)
                
            #     # 调整x轴以匹配平滑后的数据长度
            #     smoothed_x = x_values[window_size-1:]
                
            #     # 绘制平滑曲线和原始散点
            #     ax.plot(smoothed_x, smoothed, '-', linewidth=2, label='Smoothed')
            #     ax.scatter(x_values, values, alpha=0.3, label='Original')
                
            #     # 添加误差带
            #     ax.fill_between(smoothed_x, smoothed-std_values, smoothed+std_values, 
            #                 alpha=0.2, label='±1 StdDev')
            # else:
            ax.plot(x_values, values, 'o-', label='Data')
            
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

    def plot_metrics(self, pname):
        env_name = self.config['train']['env_name']
        net_name = f"{env_name}_metrics"
        plots_dir = f"{self.config['config']['output']}/{pname}"
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

    def update_metrics(self, result, sched_none_assignment_num, env, pname, output_folder, runtime):
        for key in self.metrics.keys():
            self.metrics[key].append(result[key])
        self.plot_metrics(pname)

        self.last_result = result

        if sched_none_assignment_num==0:
            if result['Total cost'] < self.best_cost:
                assignments = env.results()
                env.save(f"{output_folder}/{pname}/{pname}.json")
                out_path = f"{output_folder}/{pname}/{pname}.best_solution.xml"
                self.assignments_to_xml(assignments, pname, out_path, runtime)
                self.logger.info(f"best Episode Total cost: {result['Total cost']}")
                self.logger.info(f"best Episode Time penalty: {result['Time penalty']}")
                self.logger.info(f"best Episode Room penalty: {result['Room penalty']}")
                self.logger.info(f"best Episode Distribution penalty: {result['Distribution penalty']}")
                self.logger.info("====================================================================")
                self.best_result = result
                self.best_cost = result['Total cost']
                return True
            else:
                assignments = env.results()
                env.save(f"{output_folder}/{pname}/{pname}.json")
                out_path = f"{output_folder}/{pname}/{pname}.last_solution.xml"
                self.logger.info(f"valid Episode Total cost: {result['Total cost']}")
                self.logger.info(f"valid Episode Time penalty: {result['Time penalty']}")
                self.logger.info(f"valid Episode Room penalty: {result['Room penalty']}")
                self.logger.info(f"valid Episode Distribution penalty: {result['Distribution penalty']}")
                self.logger.info("====================================================================")
                return False
        else:
            assignments = env.results()
            env.save(f"{output_folder}/{pname}/{pname}.json")
            out_path = f"{output_folder}/{pname}/{pname}.last_solution.xml"
            self.logger.info(f"Episode Total cost: {result['Total cost']}")
            self.logger.info(f"Episode no assignment: {sched_none_assignment_num}")
            self.logger.info(f"Episode Time penalty: {result['Time penalty']}")
            self.logger.info(f"Episode Room penalty: {result['Room penalty']}")
            self.logger.info(f"Episode Distribution penalty: {result['Distribution penalty']}")
            self.logger.info("====================================================================")
            return False
    
    def conclude(self, runtime, report, pname, output_folder):
        self.logger.info("results:")
        self.logger.info(f"Total runtime: {runtime}")
        if self.best_cost < inf:
            self.logger.info(f"best Episode Total cost: {self.best_result['Total cost']}")
            self.logger.info(f"best Episode Time penalty: {self.best_result['Time penalty']}")
            self.logger.info(f"best Episode Room penalty: {self.best_result['Room penalty']}")
            self.logger.info(f"best Episode Distribution penalty: {self.best_result['Distribution penalty']}")
            if report:
                validor_result = report_result(f"{output_folder}/{pname}/{pname}.best_solution.xml")
                self.logger.info(f"Validation result: {validor_result}")
        else:
            self.logger.info(f"no valid assignments!")
            self.logger.info(f"best Episode Total cost: {self.last_result['Total cost']}")
            self.logger.info(f"last Episode Time penalty: {self.last_result['Time penalty']}")
            self.logger.info(f"last Episode Room penalty: {self.last_result['Room penalty']}")
            self.logger.info(f"last Episode Distribution penalty: {self.last_result['Distribution penalty']}")
            if report:
                validor_result = report_result(f"{output_folder}/{pname}/{pname}.last_solution.xml")
                self.logger.info(f"Validation result: {validor_result}")