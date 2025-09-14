import os
import sys
import json
import logging
import time
import yaml
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch
import argparse
import gzip

# 尝试导入 swanlab，如果失败则使用一个虚拟(Dummy)对象来兼容接口
# 这样即使没有安装 swanlab，代码也能正常运行，只是不会进行在线日志记录
try:
    import swanlab as wandb
except ImportError:
    print("警告: swanlab 未安装。将使用一个虚拟日志记录器。请运行 'pip install swanlab' 安装。")
    
    class DummyWandb:
        """
        一个虚拟的 WandB/SwanLab 类，当未安装相应库时，提供相同接口的空操作。
        这可以避免在代码中到处使用 if/else 来检查库是否存在。
        """
        def __init__(self):
            # 存储配置信息，使其行为与真实 wandb.config 类似
            self.config = {}
            # 虚拟一个 run 对象，避免访问 run.url 等属性时出错
            self.run = SimpleNamespace(url="local session (swanlab not installed)")

        def init(self, *args, **kwargs):
            """虚拟的 init 方法，记录配置信息。"""
            print("虚拟 WandB/SwanLab: init 调用被忽略。")
            # 将传入的 config 更新到 self.config 中
            if 'config' in kwargs:
                self.config.update(kwargs['config'])

        def log(self, *args, **kwargs):
            """虚拟的 log 方法，不做任何事。"""
            pass
        
        # SwanLab 使用 swanlab.Image 和 swanlab.Video
        # 这里定义虚拟类以匹配接口
        class Image:
            def __init__(self, *args, **kwargs): pass
        
        class Video:
            def __init__(self, *args, **kwargs): pass

    wandb = DummyWandb()


# 从项目中导入核心模块
from src.envs import construct_envs
from src.agent.unigoal.agent import UniGoal_Agent
from src.map.bev_mapping import BEV_Map
from src.graph.graph import Graph

class Evaluator:
    """
    将整个评估流程封装在一个类中，管理状态、模块和执行逻辑。
    """
    def __init__(self, args: SimpleNamespace):
        """
        类的构造函数，初始化所有必要的组件。
        
        Args:
            args (SimpleNamespace): 包含所有配置参数的对象。
        """
        self.args = args
        self._setup_logging_and_wandb()

        # 初始化环境、Agent 和其他核心模块
        self.envs = construct_envs(args)
        self.agent = UniGoal_Agent(args, self.envs)
        self.bev_map = BEV_Map(args)
        self.scene_graph = Graph(args)

        # 使用 deque 存储最近 N 个 episode 的指标，方便计算滑动平均
        self.episode_metrics = {
            'success': deque(maxlen=args.num_episodes),
            'spl': deque(maxlen=args.num_episodes),
            'episode_length': deque(maxlen=args.num_episodes),
        }

    def _setup_logging_and_wandb(self):
        """初始化文件日志记录器和 SwanLab/WandB 运行实例。"""
        # 创建日志和可视化结果的输出目录
        os.makedirs(self.args.log_dir, exist_ok=True)
        os.makedirs(self.args.visualization_dir, exist_ok=True)

        # 配置 Python 的 logging 模块，将日志写入文件
        logging.basicConfig(
            filename=os.path.join(self.args.log_dir, 'eval.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("="*50)
        logging.info("评估开始，加载配置参数...")
        logging.info(self.args)

        # 初始化 SwanLab/WandB
        wandb.init(
            project=self.args.project_name,
            experiment_name=self.args.experiment_id, # 使用 experiment_name 字段以兼容 swanlab
            config=vars(self.args)
        )
        print(f"SwanLab/WandB 运行已初始化。请在以下地址查看: {getattr(wandb.run, 'url', '本地会话')}")

    def _reset_episode_state(self, infos: dict, done=False):
        """
        在每个新的 episode 开始时，重置自定义模块（地图、图）的状态。
        注意：这个方法不重置环境本身（那是由 agent.reset() 完成的）。
        
        Args:
            infos (dict): 从环境中获取的包含 episode 信息的字典。
        """
        # 为新环境重置 BEV 地图和智能体位姿
        if done:
            self.bev_map.update_intrinsic_rew()
            self.bev_map.init_map_and_pose_for_env()
        
        # 重置场景图
        self.scene_graph.reset()
        self.scene_graph.set_obj_goal(infos['goal_name'])
        
        # 根据配置设置图像或文本目标
        if self.args.goal_type == 'ins-image':
            self.scene_graph.set_image_goal(infos['instance_imagegoal'])
        elif self.args.goal_type == 'text':
            self.scene_graph.set_text_goal(infos['text_goal'])
        
        # 清空上一轮的可视化图像列表
        self.agent.vis_image_list = []

    def _prepare_agent_input(self, local_step: int, global_goals: list) -> dict:
        """
        为 agent.step() 方法准备输入字典。
        此方法将所有需要的地图、位姿和目标信息组装起来。

        Args:
            local_step (int): 当前局部路径规划的步数。
            global_goals (list): 当前的全局目标点 [row, col]。

        Returns:
            dict: 准备好的输入字典。
        """
        # 创建目标地图，仅在全局目标点处值为 1
        goal_maps = np.zeros((self.args.local_width, self.args.local_height))
        goal_maps[global_goals[0], global_goals[1]] = 1
        
        # 组装输入字典
        agent_input = {
            'map_pred': self.bev_map.local_map[0, 0, :, :].cpu().numpy(),
            'exp_pred': self.bev_map.local_map[0, 1, :, :].cpu().numpy(),
            'pose_pred': self.bev_map.planner_pose_inputs[0],
            'goal': goal_maps,
            'exp_goal': goal_maps * 1,
            # 当到达局部路径终点时，标记为 new_goal
            'new_goal': local_step == self.args.num_local_steps - 1,
            'found_goal': False,  # 此处 'found_goal' 的逻辑在原代码中未被有效使用，保持默认
            'wait': False, # 重构后不再需要 wait 标志，Agent 总是活动的
            'sem_map': self.bev_map.local_map[0, 4:11, :, :].cpu().numpy(),
            'all_frontiers': None,
            'tsp_path_info': None
        }

        # 如果启用可视化，添加额外的语义地图信息
        if self.args.visualize:
            self.bev_map.local_map[0, 10, :, :] = 1e-5 # 可能是为可视化添加的标记
            agent_input['sem_map_pred'] = self.bev_map.local_map[0, 4:11, :, :].argmax(0).cpu().numpy()
            agent_input['semantic_map_vis'] = self.scene_graph.visualize_semantic_map()
            # 从图模块获取所有前沿点
            if hasattr(self.scene_graph, 'frontier_locations'):
                agent_input['all_frontiers'] = self.scene_graph.frontier_locations
            # 从图模块获取TSP规划信息
            if hasattr(self.scene_graph, 'tsp_path_info') and self.scene_graph.tsp_path_info is not None:
                agent_input['tsp_path_info'] = self.scene_graph.tsp_path_info

        return agent_input

    def run(self):
        """
        启动并执行主评估循环。
        """
        start_time = time.time()
        num_finished_episodes = 0
        total_steps = 0 # 记录所有 episode 的总步数

        print("--- 开始评估流程 ---")

        # --- 第一个 episode 的初始重置 ---
        # Agent 重置会重置环境并返回初始观测
        obs, rgbd, infos = self.agent.reset()
        # 初始化 BEV 地图和场景图
        self._reset_episode_state(infos)
        self.bev_map.init_map_and_pose()
        self.bev_map.mapping(rgbd, infos)
        
        
        # 初始全局目标设定在局部地图中心
        global_goals = [self.args.local_width // 2, self.args.local_height // 2]
        current_episode_steps = 0

        # --- 主循环 ---
        # 循环直到完成指定数量的 episodes
        while num_finished_episodes < self.args.num_episodes:
            
            # --- 1. 更新地图和图 ---
            # 使用上一步的观测数据更新 BEV 地图和场景图
            self.bev_map.mapping(rgbd, infos)
            self.scene_graph.set_full_map(self.bev_map.full_map)
            self.scene_graph.set_full_pose(self.bev_map.full_pose)
            self.scene_graph.set_navigate_steps(current_episode_steps)
            # 偶数步更新场景图中的观测（这部分逻辑来自原代码）
            if current_episode_steps % 2 == 0:
                self.scene_graph.set_observations(obs)
                self.scene_graph.update_scenegraph()

            # --- 2. 全局路径规划 ---
            # 全局目标规划在两种情况下触发：
            # 1. 局部规划周期结束 (local_step == num_local_steps - 1)
            # 2. 智能体已到达当前全局目标点附近
            local_step = current_episode_steps % self.args.num_local_steps

            # 计算智能体当前位置到全局目标的距离（都在局部地图坐标系下）
            # self.bev_map.planner_pose_inputs[0] 的前两个元素是智能体在局部地图中的 (row, col) 坐标

            distance_to_goal = np.linalg.norm(np.array([self.bev_map.local_row, self.bev_map.local_col]) - np.array(global_goals))
            
            # 定义 "到达" 阈值，10个像素约等于50cm（当分辨率为5cm/px时），这是一个合理的“足够近”的距离

            # 当局部规划周期结束 或 已到达目标点时，触发全局规划
            print(f"----(距离当前目标 {distance_to_goal})。")
            if local_step == self.args.num_local_steps - 1 or (distance_to_goal < 15 and current_episode_steps > 50):
                # 仅在实际需要移动地图时（即周期结束时）才移动
                if local_step == self.args.num_local_steps - 1:
                    self.bev_map.update_intrinsic_rew()
                    self.bev_map.move_local_map()
                
                # 如果是因为到达目标而提前触发，可以打印一条信息以便调试
                print(f"  (步数 {current_episode_steps})")
                
                # 更新图模块中的地图和位姿，然后探索新目标
                self.scene_graph.set_full_map(self.bev_map.full_map)
                self.scene_graph.set_full_pose(self.bev_map.full_pose)
                goal = self.scene_graph.explore()
                
                # 如果探索到有效目标，则更新全局目标点
                if isinstance(goal, (list, np.ndarray)):
                    # 将全局坐标系下的 goal 转换到局部地图坐标系
                    goal_x = goal[0] - self.bev_map.local_map_boundary[0, 0]
                    goal_y = goal[1] - self.bev_map.local_map_boundary[0, 2]
                    # 检查转换后的坐标是否在局部地图范围内
                    if 0 <= goal_x < self.args.local_width and 0 <= goal_y < self.args.local_height:
                        global_goals = [int(goal_x), int(goal_y)]

            # --- 3. 准备 Agent 输入并执行一步 ---
            agent_input = self._prepare_agent_input(local_step, global_goals)
            
            # 如果开启可视化 清空可视化通道
            if self.args.visualize:
                self.bev_map.local_map[0, 10, :, :] = 1e-5 
                agent_input['sem_map_pred'] = self.bev_map.local_map[0, 4:11, :, :].argmax(0).cpu().numpy()

            # 在环境中执行一步，获取新的观测
            obs, rgbd, done, infos = self.agent.step(agent_input)
            
            # 更新步数计数器
            current_episode_steps += 1
            total_steps += 1

            # --- 4. 检查 Episode 是否结束 ---
            if done:
                print(f"--- Episode {infos['episode_no']} 完成, 耗时 {current_episode_steps} 步 ---")
                
                # 记录刚刚结束的 episode 的指标
                self._log_episode_metrics(infos, current_episode_steps, num_finished_episodes)
                num_finished_episodes += 1
                # 如果还有 episode 需要运行，则为下一个进行重置
                if num_finished_episodes < self.args.num_episodes:
                    print("重置环境和状态以进行下一个 episode...")
                    # 关键步骤：调用 agent.reset() 获取新 episode 的初始观测
                    obs, rgbd, infos = self.agent.reset()
                    # 重置我们自己的内部状态跟踪器（地图、图等）
                    self._reset_episode_state(infos, done)
                    # 重置当前 episode 的步数和初始目标
                    current_episode_steps = 0
                    global_goals = [self.args.local_width // 2, self.args.local_height // 2]
        
        # --- 循环结束后 ---
        self._log_final_summary()
        total_time = time.time() - start_time
        print(f"\n评估流程全部完成，总耗时: {total_time:.2f} 秒。")
        logging.info(f"评估流程全部完成，总耗时: {total_time:.2f} 秒。")

    def _log_episode_metrics(self, infos: dict, episode_steps: int, episode_count: int):
        """
        记录单个完成的 episode 的指标，并处理可视化结果。

        Args:
            infos (dict): 环境返回的信息字典。
            episode_steps (int): 该 episode 的总步数。
            episode_count (int): 已完成的 episode 数量。
        """
        success = infos.get('success', 0.0)
        spl = infos.get('spl', 0.0)
        episode_no = infos.get('episode_no', 'N/A')

        # 存储指标
        self.episode_metrics['success'].append(success)
        self.episode_metrics['spl'].append(spl)
        self.episode_metrics['episode_length'].append(episode_steps)

        # 打印到控制台并记录到日志文件
        log_str = f"Episode[{episode_no}] | Success: {success:.2f}, SPL: {spl:.2f}, Length: {episode_steps}"
        print(log_str)
        logging.info(log_str)

        # 准备上传到 SwanLab 的数据
        metrics_to_log = {
            'Per-Episode/Success': success,
            'Per-Episode/SPL': spl,
            'Per-Episode/Length': episode_steps,
            'Per-Episode/Distance to Goal': infos.get('distance_to_goal', 0.0),
        }

        # 如果开启了可视化，仅将视频保存到本地 MP4 文件
        if self.args.visualize and self.agent.vis_image_list:
            video_dir = os.path.join(self.args.visualization_dir, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            
            # 定义 MP4 的文件路径
            base_filename = f'eps_{episode_no:0>6}'
            mp4_path = os.path.join(video_dir, f'{base_filename}.mp4')

            # agent 保存 mp4
            self.agent.save_visualization(mp4_path)
            print(f"可视化视频已保存至: {mp4_path}")


        # 将所有指标上传到 SwanLab
        wandb.log(metrics_to_log, step=episode_count)

    def _log_final_summary(self):
        """计算并记录所有 episodes 的最终平均指标。"""
        # 计算平均值，如果列表为空则默认为 0
        mean_success = np.mean(self.episode_metrics['success']) if self.episode_metrics['success'] else 0.0
        mean_spl = np.mean(self.episode_metrics['spl']) if self.episode_metrics['spl'] else 0.0
        mean_length = np.mean(self.episode_metrics['episode_length']) if self.episode_metrics['episode_length'] else 0.0
        
        # 格式化最终摘要字符串
        summary_str = (
            f"\n{'='*40}\n"
            f"最终评估摘要 ({len(self.episode_metrics['success'])} episodes):\n"
            f"  - 平均成功率 (SR)  : {mean_success:.5f}\n"
            f"  - 平均 SPL         : {mean_spl:.5f}\n"
            f"  - 平均 Episode 长度: {mean_length:.2f}\n"
            f"{'='*40}"
        )
        print(summary_str)
        logging.info(summary_str)

        # 将最终摘要上传到 SwanLab
        wandb.log({
            'Aggregate/Average_Success_Rate': mean_success,
            'Aggregate/Average_SPL': mean_spl,
            'Aggregate/Average_Episode_Length': mean_length,
        })
        
        # 将最终结果保存为 JSON 文件，便于后续分析
        results_path = os.path.join(self.args.log_dir, 'summary_results.json')
        # 移除无法序列化的对象（如 torch.device）
        config_for_json = {k: v for k, v in vars(self.args).items() if not isinstance(v, torch.device)}
        final_results = {
            'config': config_for_json,
            'metrics': {
                'success_rate': mean_success,
                'spl': mean_spl,
                'episode_length': mean_length,
                'raw_success': list(self.episode_metrics['success']),
                'raw_spl': list(self.episode_metrics['spl']),
            }
        }
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"最终结果已保存至: {results_path}")


def get_config() -> SimpleNamespace:
    """
    解析命令行参数和 YAML 配置文件，并整合成一个配置对象。
    """
    parser = argparse.ArgumentParser(description="基于 UniGoal 的视觉导航评估脚本")
    
    # 核心参数
    parser.add_argument("--config-file", default="configs/config_habitat.yaml", metavar="FILE", help="配置文件的路径")
    parser.add_argument("--project-name", default="unigoal-paper", type=str, help="用于 SwanLab/WandB 日志记录的项目名称")
    parser.add_argument("--goal_type", default="ins-image", type=str, choices=['ins-image', 'text'], help="目标类型")
    
    # 可选的运行时参数
    parser.add_argument("--episode_id", default=-1, type=int, help="指定运行单个 episode 的 ID (默认-1, 运行所有)")
    parser.add_argument("--goal", default="", type=str, help="（保留）指定目标的字符串")
    parser.add_argument("--real_world", action="store_true", help="是否在真实世界环境中运行的标志")
    
    args = parser.parse_args()

    # 加载 YAML 配置文件
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # 将 YAML 配置内容更新到 args 中，命令行参数优先级更高
    args_dict = vars(args)
    args_dict.update(config)
    args = SimpleNamespace(**args_dict)

    # 调试模式检测
    args.is_debugging = sys.gettrace() is not None
    if args.is_debugging:
        args.experiment_id = "debug"
        print("--- 检测到调试模式，实验ID设置为 'debug' ---")

    # 根据配置计算派生参数
    args.log_dir = os.path.join(args.dump_location, args.experiment_id, 'log')
    args.visualization_dir = os.path.join(args.dump_location, args.experiment_id, 'visualization')
    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width, args.global_height = args.map_size, args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    args.num_scenes = args.num_processes
    args.num_episodes = int(args.num_eval_episodes)
    
    return args


def main():
    """脚本主入口。"""
    # 1. 获取配置
    args = get_config()
    # 2. 创建评估器实例
    evaluator = Evaluator(args)
    # 3. 运行评估
    evaluator.run()


if __name__ == "__main__":
    main()