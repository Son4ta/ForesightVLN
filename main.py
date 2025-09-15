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

# Try to import swanlab, if it fails, use a dummy object for interface compatibility
# This way the code can run normally even without swanlab installed, just without online logging
try:
    import swanlab as wandb
except ImportError:
    print("Warning: swanlab not installed. Will use a dummy logger. Please run 'pip install swanlab' to install.")
    
    class DummyWandb:
        """
        A dummy WandB/SwanLab class that provides no-op operations with the same interface
        when the corresponding library is not installed.
        This avoids having to use if/else checks throughout the code to check if the library exists.
        """
        def __init__(self):
            # Store configuration information to behave similarly to real wandb.config
            self.config = {}
            # Create a dummy run object to avoid errors when accessing run.url and other attributes
            self.run = SimpleNamespace(url="local session (swanlab not installed)")

        def init(self, *args, **kwargs):
            """Dummy init method that records configuration information."""
            print("Dummy WandB/SwanLab: init call ignored.")
            # Update self.config with the passed config
            if 'config' in kwargs:
                self.config.update(kwargs['config'])

        def log(self, *args, **kwargs):
            """Dummy log method that does nothing."""
            pass
        
        # SwanLab uses swanlab.Image and swanlab.Video
        # Define dummy classes here to match the interface
        class Image:
            def __init__(self, *args, **kwargs): pass
        
        class Video:
            def __init__(self, *args, **kwargs): pass

    wandb = DummyWandb()


# Import core modules from the project
from src.envs import construct_envs
from src.agent.unigoal.agent import UniGoal_Agent
from src.map.bev_mapping import BEV_Map
from src.graph.graph import Graph

class Evaluator:
    """
    Encapsulates the entire evaluation process in a class, managing state, modules, and execution logic.
    """
    def __init__(self, args: SimpleNamespace):
        """
        Class constructor that initializes all necessary components.
        
        Args:
            args (SimpleNamespace): Object containing all configuration parameters.
        """
        self.args = args
        self._setup_logging_and_wandb()

        # Initialize environment, Agent and other core modules
        self.envs = construct_envs(args)
        self.agent = UniGoal_Agent(args, self.envs)
        self.bev_map = BEV_Map(args)
        self.scene_graph = Graph(args)

        # Use deque to store metrics for the last N episodes, convenient for calculating moving averages
        self.episode_metrics = {
            'success': deque(maxlen=args.num_episodes),
            'spl': deque(maxlen=args.num_episodes),
            'episode_length': deque(maxlen=args.num_episodes),
        }

    def _setup_logging_and_wandb(self):
        """Initialize file logger and SwanLab/WandB run instance."""
        # Create output directories for logs and visualization results
        os.makedirs(self.args.log_dir, exist_ok=True)
        os.makedirs(self.args.visualization_dir, exist_ok=True)

        # Configure Python's logging module to write logs to file
        logging.basicConfig(
            filename=os.path.join(self.args.log_dir, 'eval.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("="*50)
        logging.info("Evaluation started, loading configuration parameters...")
        logging.info(self.args)

        # Initialize SwanLab/WandB
        wandb.init(
            project=self.args.project_name,
            experiment_name=self.args.experiment_id, # Use experiment_name field for swanlab compatibility
            config=vars(self.args)
        )
        print(f"SwanLab/WandB run initialized. Please check at: {getattr(wandb.run, 'url', 'local session')}")

    def _reset_episode_state(self, infos: dict, done=False):
        """
        Reset the state of custom modules (map, graph) at the beginning of each new episode.
        Note: This method does not reset the environment itself (that is done by agent.reset()).
        
        Args:
            infos (dict): Dictionary containing episode information obtained from the environment.
        """
        # Reset BEV map and agent pose for new environment
        if done:
            self.bev_map.update_intrinsic_rew()
            self.bev_map.init_map_and_pose_for_env()
        
        # Reset scene graph
        self.scene_graph.reset()
        self.scene_graph.set_obj_goal(infos['goal_name'])
        
        # Set image or text goal based on configuration
        if self.args.goal_type == 'ins-image':
            self.scene_graph.set_image_goal(infos['instance_imagegoal'])
        elif self.args.goal_type == 'text':
            self.scene_graph.set_text_goal(infos['text_goal'])
        
        # Clear visualization image list from previous round
        self.agent.vis_image_list = []

    def _prepare_agent_input(self, local_step: int, global_goals: list) -> dict:
        """
        Prepare input dictionary for agent.step() method.
        This method assembles all necessary map, pose and goal information.

        Args:
            local_step (int): Current number of steps in local path planning.
            global_goals (list): Current global goal point [row, col].

        Returns:
            dict: Prepared input dictionary.
        """
        # Create goal map with value 1 only at global goal points
        goal_maps = np.zeros((self.args.local_width, self.args.local_height))
        goal_maps[global_goals[0], global_goals[1]] = 1
        
        # Assemble input dictionary
        agent_input = {
            'map_pred': self.bev_map.local_map[0, 0, :, :].cpu().numpy(),
            'exp_pred': self.bev_map.local_map[0, 1, :, :].cpu().numpy(),
            'pose_pred': self.bev_map.planner_pose_inputs[0],
            'goal': goal_maps,
            'exp_goal': goal_maps * 1,
            # Mark as new_goal when reaching the end of local path
            'new_goal': local_step == self.args.num_local_steps - 1,
            'found_goal': False,  # The logic for 'found_goal' was not effectively used in the original code, keeping default
            'wait': False, # After refactoring, wait flag is no longer needed, Agent is always active
            'sem_map': self.bev_map.local_map[0, 4:11, :, :].cpu().numpy(),
            'all_frontiers': None,
            'tsp_path_info': None
        }

        # If visualization is enabled, add additional semantic map information
        if self.args.visualize:
            self.bev_map.local_map[0, 10, :, :] = 1e-5 # Possibly a marker added for visualization
            agent_input['sem_map_pred'] = self.bev_map.local_map[0, 4:11, :, :].argmax(0).cpu().numpy()
            agent_input['semantic_map_vis'] = self.scene_graph.visualize_semantic_map()
            # Get all frontier points from graph module
            if hasattr(self.scene_graph, 'frontier_locations'):
                agent_input['all_frontiers'] = self.scene_graph.frontier_locations
            # Get TSP planning information from graph module
            if hasattr(self.scene_graph, 'tsp_path_info') and self.scene_graph.tsp_path_info is not None:
                agent_input['tsp_path_info'] = self.scene_graph.tsp_path_info

        return agent_input

    def run(self):
        """
        Start and execute the main evaluation loop.
        """
        start_time = time.time()
        num_finished_episodes = 0
        total_steps = 0 # Record total steps for all episodes

        print("--- Starting evaluation process ---")

        # --- Initial reset for first episode ---
        # Agent reset will reset environment and return initial observations
        obs, rgbd, infos = self.agent.reset()
        # Initialize BEV map and scene graph
        self._reset_episode_state(infos)
        self.bev_map.init_map_and_pose()
        self.bev_map.mapping(rgbd, infos)
        
        
        # Initial global goal set at local map center
        global_goals = [self.args.local_width // 2, self.args.local_height // 2]
        current_episode_steps = 0

        # --- Main loop ---
        # Loop until specified number of episodes is completed
        while num_finished_episodes < self.args.num_episodes:
            
            # --- 1. Update map and graph ---
            # Update BEV map and scene graph using observation data from previous step
            self.bev_map.mapping(rgbd, infos)
            self.scene_graph.set_full_map(self.bev_map.full_map)
            self.scene_graph.set_full_pose(self.bev_map.full_pose)
            self.scene_graph.set_navigate_steps(current_episode_steps)
            # Update observations in scene graph at even steps (this logic comes from original code)
            if current_episode_steps % 2 == 0:
                self.scene_graph.set_observations(obs)
                self.scene_graph.update_scenegraph()

            # --- 2. Global path planning ---
            # Global goal planning is triggered in two cases:
            # 1. Local planning cycle ends (local_step == num_local_steps - 1)
            # 2. Agent has reached near the current global goal point
            local_step = current_episode_steps % self.args.num_local_steps

            # Calculate distance from agent's current position to global goal (both in local map coordinate system)
            # The first two elements of self.bev_map.planner_pose_inputs[0] are the agent's (row, col) coordinates in local map

            distance_to_goal = np.linalg.norm(np.array([self.bev_map.local_row, self.bev_map.local_col]) - np.array(global_goals))
            
            # Define "arrival" threshold, 10 pixels approximately equals 50cm (when resolution is 5cm/px), this is a reasonable "close enough" distance

            # Trigger global planning when local planning cycle ends or goal point is reached
            print(f"----(Distance to current goal {distance_to_goal}).")
            if local_step == self.args.num_local_steps - 1 or (distance_to_goal < 15 and current_episode_steps > 50):
                # Only move when actually need to move map (i.e., at cycle end)
                if local_step == self.args.num_local_steps - 1:
                    self.bev_map.update_intrinsic_rew()
                    self.bev_map.move_local_map()
                
                # If triggered early due to reaching goal, print a message for debugging
                print(f"  (Steps {current_episode_steps})")
                
                # Update map and pose in graph module, then explore new goals
                self.scene_graph.set_full_map(self.bev_map.full_map)
                self.scene_graph.set_full_pose(self.bev_map.full_pose)
                goal = self.scene_graph.explore()
                
                # If valid goal is explored, update global goal point
                if isinstance(goal, (list, np.ndarray)):
                    # Convert goal from global coordinate system to local map coordinate system
                    goal_x = goal[0] - self.bev_map.local_map_boundary[0, 0]
                    goal_y = goal[1] - self.bev_map.local_map_boundary[0, 2]
                    # Check if converted coordinates are within local map range
                    if 0 <= goal_x < self.args.local_width and 0 <= goal_y < self.args.local_height:
                        global_goals = [int(goal_x), int(goal_y)]

            # --- 3. Prepare Agent input and execute one step ---
            agent_input = self._prepare_agent_input(local_step, global_goals)
            
            # If visualization is enabled, clear visualization channel
            if self.args.visualize:
                self.bev_map.local_map[0, 10, :, :] = 1e-5 
                agent_input['sem_map_pred'] = self.bev_map.local_map[0, 4:11, :, :].argmax(0).cpu().numpy()

            # Execute one step in environment, get new observations
            obs, rgbd, done, infos = self.agent.step(agent_input)
            
            # Update step counters
            current_episode_steps += 1
            total_steps += 1

            # --- 4. Check if Episode is finished ---
            if done:
                print(f"--- Episode {infos['episode_no']} completed, took {current_episode_steps} steps ---")
                
                # Record metrics for the just completed episode
                self._log_episode_metrics(infos, current_episode_steps, num_finished_episodes)
                num_finished_episodes += 1
                # If there are more episodes to run, reset for the next one
                if num_finished_episodes < self.args.num_episodes:
                    print("Resetting environment and state for next episode...")
                    # Key step: call agent.reset() to get initial observations for new episode
                    obs, rgbd, infos = self.agent.reset()
                    # Reset our own internal state trackers (map, graph, etc.)
                    self._reset_episode_state(infos, done)
                    # Reset current episode step count and initial goal
                    current_episode_steps = 0
                    global_goals = [self.args.local_width // 2, self.args.local_height // 2]
        
        # --- After loop ends ---
        self._log_final_summary()
        total_time = time.time() - start_time
        print(f"\nEvaluation process completed, total time: {total_time:.2f} seconds.")
        logging.info(f"Evaluation process completed, total time: {total_time:.2f} seconds.")

    def _log_episode_metrics(self, infos: dict, episode_steps: int, episode_count: int):
        """
        Record metrics for a single completed episode and handle visualization results.

        Args:
            infos (dict): Information dictionary returned by environment.
            episode_steps (int): Total steps for this episode.
            episode_count (int): Number of completed episodes.
        """
        success = infos.get('success', 0.0)
        spl = infos.get('spl', 0.0)
        episode_no = infos.get('episode_no', 'N/A')

        # Store metrics
        self.episode_metrics['success'].append(success)
        self.episode_metrics['spl'].append(spl)
        self.episode_metrics['episode_length'].append(episode_steps)

        # Print to console and record to log file
        log_str = f"Episode[{episode_no}] | Success: {success:.2f}, SPL: {spl:.2f}, Length: {episode_steps}"
        print(log_str)
        logging.info(log_str)

        # Prepare data to upload to SwanLab
        metrics_to_log = {
            'Per-Episode/Success': success,
            'Per-Episode/SPL': spl,
            'Per-Episode/Length': episode_steps,
            'Per-Episode/Distance to Goal': infos.get('distance_to_goal', 0.0),
        }

        # If visualization is enabled, save video to local MP4 file only
        if self.args.visualize and self.agent.vis_image_list:
            video_dir = os.path.join(self.args.visualization_dir, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            
            # Define MP4 file path
            base_filename = f'eps_{episode_no:0>6}'
            mp4_path = os.path.join(video_dir, f'{base_filename}.mp4')

            # agent saves mp4
            self.agent.save_visualization(mp4_path)
            print(f"Visualization video saved to: {mp4_path}")


        # Upload all metrics to SwanLab
        wandb.log(metrics_to_log, step=episode_count)

    def _log_final_summary(self):
        """Calculate and record final average metrics for all episodes."""
        # Calculate averages, default to 0 if list is empty
        mean_success = np.mean(self.episode_metrics['success']) if self.episode_metrics['success'] else 0.0
        mean_spl = np.mean(self.episode_metrics['spl']) if self.episode_metrics['spl'] else 0.0
        mean_length = np.mean(self.episode_metrics['episode_length']) if self.episode_metrics['episode_length'] else 0.0
        
        # Format final summary string
        summary_str = (
            f"\n{'='*40}\n"
            f"Final evaluation summary ({len(self.episode_metrics['success'])} episodes):\n"
            f"  - Average Success Rate (SR): {mean_success:.5f}\n"
            f"  - Average SPL              : {mean_spl:.5f}\n"
            f"  - Average Episode Length   : {mean_length:.2f}\n"
            f"{'='*40}"
        )
        print(summary_str)
        logging.info(summary_str)

        # Upload final summary to SwanLab
        wandb.log({
            'Aggregate/Average_Success_Rate': mean_success,
            'Aggregate/Average_SPL': mean_spl,
            'Aggregate/Average_Episode_Length': mean_length,
        })
        
        # Save final results as JSON file for subsequent analysis
        results_path = os.path.join(self.args.log_dir, 'summary_results.json')
        # Remove non-serializable objects (like torch.device)
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
        print(f"Final results saved to: {results_path}")


def get_config() -> SimpleNamespace:
    """
    Parse command line arguments and YAML configuration file, and integrate into a configuration object.
    """
    parser = argparse.ArgumentParser(description="UniGoal-based visual navigation evaluation script")
    
    # Core parameters
    parser.add_argument("--config-file", default="configs/config_habitat.yaml", metavar="FILE", help="配置文件的路径")
    parser.add_argument("--project-name", default="unigoal-paper", type=str, help="Project name for SwanLab/WandB logging")
    parser.add_argument("--goal_type", default="ins-image", type=str, choices=['ins-image', 'text'], help="Goal type")
    
    # Optional runtime parameters
    parser.add_argument("--episode_id", default=-1, type=int, help="Specify ID of single episode to run (default -1, run all)")
    parser.add_argument("--goal", default="", type=str, help="(Reserved) String specifying the goal")
    parser.add_argument("--real_world", action="store_true", help="Flag indicating whether to run in real world environment")
    
    args = parser.parse_args()

    # Load YAML configuration file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update YAML configuration content to args, command line arguments have higher priority
    args_dict = vars(args)
    args_dict.update(config)
    args = SimpleNamespace(**args_dict)

    # Debug mode detection
    args.is_debugging = sys.gettrace() is not None
    if args.is_debugging:
        args.experiment_id = "debug"
        print("--- Debug mode detected, experiment ID set to 'debug' ---")

    # Calculate derived parameters based on configuration
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
    """Main entry point of the script."""
    # 1. Get configuration
    args = get_config()
    # 2. Create evaluator instance
    evaluator = Evaluator(args)
    # 3. Run evaluation
    evaluator.run()


if __name__ == "__main__":
    main()