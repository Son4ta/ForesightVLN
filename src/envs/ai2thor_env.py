# src/envs/ai2thor_env.py

import ai2thor.controller
from ai2thor.platform import CloudRendering
from types import SimpleNamespace
import numpy as np
import quaternion
import random
import math
from src.utils.fmm.pose_utils import get_rel_pose_change

# 尝试从项目中导入类别映射，如果失败则使用一个默认值
try:
    from configs.categories import name2index, hm3d_to_coco
    # 你可能需要为 ProcTHOR 创建一个新的映射
    # from configs.categories import procthor_to_unigoal_map
except ImportError:
    print("Warning: Could not import category mappings from 'configs.categories'. Using defaults.")
    name2index = {}


class AI2Thor_Env:
    """
    AI2-THOR Environment Wrapper for UniGoal.
    This class mimics the interface of InstanceImageGoal_Env and uses AI2-THOR
    with the ProcTHOR dataset.
    """
    def __init__(self, args, config_env=None, dataset=None):
        """
        Initializes the AI2-THOR environment.
        Args:
            args: A namespace object containing arguments.
            config_env: Environment configuration (not used for AI2-THOR, kept for compatibility).
            dataset: A list of house JSON objects from the ProcTHOR dataset.
        """
        self.args = args
        self.dataset = dataset
        self.episode_iterator = iter(self.dataset) if self.dataset else None
        self.name2index = name2index

        # --- Action Mapping ---
        self._action_mapping = {
            0: "Stop",
            1: "MoveAhead",
            2: "RotateLeft",
            3: "RotateRight",
        }
        # AI2-THOR rotation is by default 90 degrees. If different, need to specify.
        self.rotation_degrees = self.args.rotation_degrees
        self.episode_over = False

        # --- AI2-THOR Controller Initialization ---
        self.controller = ai2thor.controller.Controller(
            agentMode="default",
            visibilityDistance=self.args.visibility_distance, # 1.5 in Habitat
            scene='FloorPlan1', # Placeholder, will be overwritten by reset
            gridSize=self.args.grid_size, # 0.25 in Habitat
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            # 输出长宽
            width=self.args.env_frame_width,
            height=self.args.env_frame_height,
            fieldOfView=self.args.hfov,
            horizon=0.0, # Look straight ahead
            # ↓为服务器端运行添加 headless 模式
            platform=CloudRendering
        )

        # --- Episode State and Metrics ---
        self.info = {}
        self.current_episode = None
        self.target_object = None
        self.start_agent_state = None
        self.goal_position = None
        self.path_length = 0.0
        self.shortest_path_length = 0.0
        self.success = False
        self.episode_over = False
        
        # Add last location tracker for sensor_pose calculation
        self.last_agent_location = None
        self.last_agent_action = None

        # Success definition
        self.success_distance = 1.0
        


    def reset(self):
        """
        Resets the environment for a new episode.
        - Loads a new house from the dataset.
        - Sets up a new goal.
        - Calculates the shortest path for SPL metric.
        - Returns the initial observation dict.
        """
        # --- Get next episode from the dataset ---
        if self.episode_iterator is None:
            raise ValueError("No dataset provided to the AI2Thor_Env.")
        try:
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
            self.current_episode = next(self.episode_iterator)
        except StopIteration:
            print("AI2-THOR dataset iterator exhausted. Restarting.")
            self.episode_iterator = iter(self.dataset)
            self.current_episode = next(self.episode_iterator)

        # --- Reset the simulator with the new scene ---
        self.controller.reset(scene=self.current_episode)
        
        # --- Randomly place agent ---
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
        start_position = random.choice(reachable_positions)
        start_rotation = random.choice([0, 90, 180, 270])
        event = self.controller.step(
            action="Teleport",
            position=start_position,
            rotation=start_rotation,
            horizon=0.0, # Look straight ahead
            standing=False
        )

        self.start_agent_state = event.metadata['agent']
        self.last_agent_location = self._get_agent_location()
        
        # --- Initialize metrics ---
        self.path_length = 0.0
        self.episode_over = False
        self.success = False
        self.last_agent_action = None

        # --- 配合一下上级调用的需要，更新goal_object_id object_category ---
        self._setup_episode_goal()

        # Calculate shortest path for SPL
        try:
            path_event = self.controller.step(
                action="GetShortestPath",
                position=self.goal_position,
                # In AI2THOR, GetShortestPath does not require objectId if a position is given.
                # However, providing it might help if there are multiple paths.
                # Since we teleport to a specific position, this should be fine.
            )
            # Check if the path was found and is valid
            if path_event.metadata['lastActionSuccess']:
                 self.shortest_path_length = path_event.metadata['actionReturn']['path_length']
            else:
                 self.shortest_path_length = float('inf')

        except Exception:
            # If path is not found (e.g., goal is unreachable), set a high value
            self.shortest_path_length = float('inf')

        obs = self._format_observation(event)
        
        # Initialize sensor_pose for the first step
        self.info['sensor_pose'] = [0., 0., 0.]

        return obs

    def step(self, action):
        """
        Executes an action in the environment.
        """
        if isinstance(action, dict):
            action_id = action['action']
        else:
            action_id = action
        # 直接从映射中获取正确的动作名 ("MoveAhead", "RotateLeft", etc.)
        action_str = self._action_mapping.get(action_id)
        
        # Store location before action for dx, dy, do calculation
        print(action_str)
        self.last_agent_action = action_str
        if action_str == "Stop":
            self.episode_over = True
            event = self.controller.last_event
        elif action_str == "MoveAhead":
            # MoveAhead action does not require additional parameters
            event = self.controller.step(action=action_str)
        else:
            '''
            根据 AI2-THOR 的官方文档，RotateLeft 和 RotateRight 默认旋转一个固定的角度
            要以一个自定义的角度旋转，应该使用 Rotate 动作，并通过 degrees 参数来指定旋转的角度
            正值为右转，负值为左转
            '''
            event = self.controller.step(action=action_str, degrees=self.rotation_degrees)
        
        # Update path length
        curr_pos = event.metadata['agent']['position']
        prev_pos = self.start_agent_state['position'] if self.path_length == 0.0 else self.controller.last_event.metadata['agent']['position']
        self.path_length += math.sqrt(
            (curr_pos['x'] - prev_pos['x'])**2 +
            (curr_pos['z'] - prev_pos['z'])**2
        )

        # Check for success
        self.success = self._is_successful(event)
        if self.success:
            self.episode_over = True

        if not event.metadata['lastActionSuccess']:
            # Optional: end episode on failed action
            # self.done = True
            pass
            
        obs = self._format_observation(event)

        # Update info with relative pose change
        # self.info['sensor_pose'] =  self._get_location_change()

        return obs #, self.episode_over, self.info
    
    def get_agent_pose(self):
        agent_meta = self.controller.last_event.metadata['agent']
        pos = agent_meta['position']
        rot_deg = agent_meta['rotation']['y']
        
        # 将AI2-THOR的左手坐标系位置 (x, y, z) 转换为Habitat的右手坐标系 (x, y, -z)
        # 这一步对于点云投影非常关键
        habitat_position = np.array([pos['x'], pos['y'], -pos['z']], dtype=np.float32)

        # AI2-THOR的旋转（左手系，y-up）和Habitat（右手系，y-up）的yaw角定义一致
        yaw_rad = np.deg2rad(-rot_deg)
        q = quaternion.from_euler_angles(0, yaw_rad, 0)

        return {'position': habitat_position, 'rotation': q}

    def _get_agent_location(self):
        # 这个函数定义了项目的“地图坐标系”
        # 它从3D世界坐标生成一个2D地图坐标 (map_x, map_y) 和朝向
        agent_pose_hab = self.get_agent_pose()
        # pos_hab 是 Habitat 坐标系下的 (x, y, z)
        pos_hab = agent_pose_hab['position']
        
        # 将3D坐标转换为2D地图坐标
        # x_map = -z_hab, y_map = -x_hab
        map_x = -pos_hab[2]
        map_y = -pos_hab[0]
        
        yaw_rad = quaternion.as_euler_angles(agent_pose_hab['rotation'])[1]
        
        return map_x, map_y, yaw_rad

    def _get_location_change(self):
        # ===============================================
        #  >>> 里程计计算 <<<
        #  计算相对位移和旋转变化，x+是向前，r+是向左旋转，r-是向右旋转
        #  注意: AI2-THOR 的坐标系是左手系
        # ===============================================
        dx, dy, do = 0.0, 0.0, 0.0
        current_location = self._get_agent_location()
        action_str = self.last_agent_action
        if action_str == None:
            # 如果没有动作，返回零变化
            return [0.0, 0.0, 0.0]
        elif action_str == "MoveAhead":
            # 获取上一帧和当前帧在地图坐标系下的位姿
            x1, y1, o1 = self.last_agent_location
            x2, y2, o2 = current_location
            dx = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # 向前移动，就是向前移动了一个固定的步长
            # dx = self.args.grid_size
        elif action_str == "RotateLeft":
            # 左转，就是逆时针旋转了一个固定的角度
            do = np.deg2rad(self.rotation_degrees)
        elif action_str == "RotateRight":
            # 右转，就是顺时针旋转了一个固定的角度
            do = -np.deg2rad(self.rotation_degrees)
        
        # [向前位移, 侧向位移, 逆时针旋转]
        self.last_agent_location = current_location
        return dx, dy, do
        
        current_location = self._get_agent_location()
        x1, y1, o1 = self.last_agent_location
        x2, y2, o2 = current_location
        
        dx, dy, do = get_rel_pose_change(
            current_location, self.last_agent_location)
        self.last_agent_location = current_location

        # 计算世界坐标系下的位移
        dx_world = x2 - x1
        dy_world = y2 - y1

        # 将世界位移旋转到上一步的局部坐标系中
        # sin(-o1) = -sin(o1), cos(-o1) = cos(o1)
        dx_local = dx_world * np.cos(o1) + dy_world * np.sin(o1)
        dy_local = -dx_world * np.sin(o1) + dy_world * np.cos(o1)
        
        # 计算朝向变化
        do = o2 - o1
        # 将角度变化归一化到 [-pi, pi]
        if do > np.pi:
            do -= 2 * np.pi
        if do < -np.pi:
            do += 2 * np.pi

        self.last_agent_location = current_location

        # UniGoal 的建图模块期望的 sensor_pose 是 [向前移动距离, 侧向移动距离, 逆时针旋转角度]
        # dx_local 是向前, dy_local 是向右, do 是逆时针旋转
        # 注意: dx_local 是米，而BEV模块的pose是米。单位一致。
        return [dx_local, dy_local, do]

    
    def get_metrics(self):
        """
        Calculates and returns navigation metrics.
        """
        dist_to_goal = float('inf')
        if self.goal_position:
            agent_pos = self.controller.last_event.metadata['agent']['position']
            dist_to_goal = math.sqrt(
                (agent_pos['x'] - self.goal_position['x'])**2 +
                (agent_pos['z'] - self.goal_position['z'])**2
            )

        spl = 0.0
        if self.success:
            if self.shortest_path_length != float('inf') and self.shortest_path_length > 0:
                spl = self.shortest_path_length / max(self.shortest_path_length, self.path_length)

        return {
            'distance_to_goal': dist_to_goal,
            'success': float(self.success),
            'spl': spl,
            'soft_spl': 0.0 # Placeholder for soft_spl
        }

    def _setup_episode_goal(self):
        """
        Selects a random object as a goal and generates goal info, including the goal image.
        """
        if not self.controller.last_event.metadata.get('objects'):
            raise RuntimeError("No objects found in the current ProcTHOR house.")
            
        # Select a valid, visible object as the target
        self.target_object = random.choice(self.controller.last_event.metadata['objects'])
        self.goal_position = self.target_object['position']
        
        # --- Populate info to self.current_episode by fake_episode---
        # 创建一个空的 SimpleNamespace 对象，因为没法直接改变 self.current_episode
        fake_episode = SimpleNamespace()
        #TODO 这里有很大问题，asset_id = 'Wall_Decor_Photo_6'，下面逻辑要重写
        asset_id = self.target_object['assetId']
        category_name = self.target_object['objectType'].lower() # Use objectType directly
        fake_episode.object_category = category_name
        # AI2-THOR does not use object IDs like Habitat, so we use its own unique id
        # fake_episode.goal_object_id = self.target_object['objectId']
        fake_episode.goal_object_id = -1 # Placeholder

        # --- Generate Goal Image for 'ins-image' task ---
        if self.args.goal_type == 'ins-image':
            goal_image = self._get_goal_image()
            self.info['instance_imagegoal'] = goal_image
            fake_episode.instance_imagegoal = goal_image
        
        if self.args.goal_type == 'text':
            self.info['text_goal'] = f"Find the {category_name.replace('_', ' ')}."
            fake_episode.text_goal = f"Find the {category_name.replace('_', ' ')}."

        #TODO ☠这里改变了self.current_episode，预计没有影响，因为后续流程没有用到current_episode，请验证
        self.current_episode = fake_episode



    def _get_goal_image(self):
        """
        在场景中从一个预定义的目标列表中寻找一个物体实例，并为其生成一个清晰的、
        且保持最小安全距离的图像。该过程不移动主Agent。
        """
        # 1. 定义一个目标物体类别列表
        # 函数将从以下列表中随机寻找一个类型的物体
        TARGET_OBJECT_TYPES = [
            "Chair", "Sofa", "ArmChair", "Bed", "Desk"
        ]

        # 2. 在场景中寻找列表内任一类型的实例
        all_objects = self.controller.last_event.metadata['objects']
        
        # 筛选出所有属于目标列表中的物体实例
        possible_targets = [obj for obj in all_objects if obj['objectType'] in TARGET_OBJECT_TYPES]
        
        if not possible_targets:
            print(f"错误: 在场景中没有找到列表 {TARGET_OBJECT_TYPES} 中的任何物体。")
            return np.zeros((self.args.frame_height, self.args.frame_width, 3), dtype=np.uint8)

        # 从找到的实例中随机选择一个
        target_object_info = random.choice(possible_targets)
        goal_position = target_object_info['position']
        print(f"已选择目标实例: {target_object_info['objectId']} (类型: {target_object_info['objectType']})")

        # 3. 寻找一个满足最小距离要求的最佳相机视点
        try:
            reachable_pos = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
            if not reachable_pos:
                raise ValueError("在环境中没有找到可到达的位置。")
        except Exception as e:
            print(f"获取可到达位置时出错: {e}")
            return np.zeros((self.args.frame_height, self.args.frame_width, 3), dtype=np.uint8)

        # ===================================================================
        # >>> 核心修改: 筛选出所有与目标距离 >= 0.5米 的可到达点 <<<
        # ===================================================================
        MIN_DISTANCE = 0.5
        valid_viewpoints = []
        for pos in reachable_pos:
            dist = math.sqrt(
                (pos['x'] - goal_position['x'])**2 +
                (pos['z'] - goal_position['z'])**2
            )
            if dist >= MIN_DISTANCE:
                valid_viewpoints.append({'pos': pos, 'dist': dist})
                
        if not valid_viewpoints:
            print(f"警告: 找不到任何与目标距离至少 {MIN_DISTANCE}米 的可到达视点。")
            return np.zeros((self.args.frame_height, self.args.frame_width, 3), dtype=np.uint8)

        # 从满足条件的视点中，找到离目标最近的一个
        best_viewpoint_info = min(valid_viewpoints, key=lambda x: x['dist'])
        best_viewpoint = best_viewpoint_info['pos']
        print(f"已找到最佳视点，距离目标 {best_viewpoint_info['dist']:.2f}米")

        # 4. 计算相机位置和旋转，使其朝向目标
        camera_position = {'x': best_viewpoint['x'], 'y': best_viewpoint['y'] + 0.6, 'z': best_viewpoint['z']}
        direction_vector = {'x': goal_position['x'] - camera_position['x'], 'z': goal_position['z'] - camera_position['z']}
        yaw_angle = math.degrees(math.atan2(direction_vector['x'], direction_vector['z']))
        pitch_angle = 30.0
        camera_rotation = {'x': pitch_angle, 'y': yaw_angle, 'z': 0}

        # 5. 添加第三人称摄像头并捕获图像
        event = self.controller.step(
            action="AddThirdPartyCamera",
            position=camera_position,
            rotation=camera_rotation,
            fieldOfView=105
        )

        if not event.third_party_camera_frames or len(event.third_party_camera_frames) == 0:
            print("错误: 添加第三人称摄像头后未能获取图像。")
            return np.zeros((self.args.frame_height, self.args.frame_width, 3), dtype=np.uint8)

        # 6. 处理图像格式并返回
        rgba_frame = event.third_party_camera_frames[0]
        rgb_frame = rgba_frame[:, :, :3].astype(np.uint8)

        print(f"成功为 {target_object_info['objectId']} 生成目标图像。")
        return rgb_frame

    def _format_observation(self, event):
        # ===============================================
        #  >>> 核心修复 3: 简化并校正传感器模拟 <<<
        #  确保深度图和坐标系与建图模块的期望一致
        # ===============================================
        obs = {}
        obs['rgb'] = event.frame.astype(np.uint8)
        
        # 1. 深度图处理：过滤无效值并转换为厘米
        depth_meters = event.depth_frame.astype(np.float32)
        depth_meters[depth_meters >= self.args.max_depth] = 0 # 过滤远距离点
        depth_meters[depth_meters <= self.args.min_depth] = 0 # 过滤近距离点
        obs['depth'] = np.expand_dims(depth_meters * 100.0, axis=2) # 转换为厘米

        # 2. GPS 和 Compass
        # get_agent_pose() 已经处理了坐标系变换
        agent_pose_hab = self.get_agent_pose()
        obs['gps'] = agent_pose_hab['position'][[2, 0]] # [-z, -x] -> [-z, x] if needed. Let's use [-z, x]
        obs['gps'][1] *= -1.0
        
        yaw_rad = quaternion.as_euler_angles(agent_pose_hab['rotation'])[1]
        obs['compass'] = np.array([yaw_rad], dtype=np.float32)

        # ===============================================
        #  >>> 在这里添加 instance_imagegoal <<<
        # ===============================================
        if self.args.goal_type == 'ins-image' and hasattr(self, 'info') and 'instance_imagegoal' in self.info:
            obs['instance_imagegoal'] = self.info['instance_imagegoal']
        
        return obs

    def _is_successful(self, event):
        """
        Checks if the agent has successfully reached the goal.
        Success is defined as being within a certain distance AND the object being visible.
        """
        dist_to_goal = math.sqrt(
            (event.metadata['agent']['position']['x'] - self.goal_position['x'])**2 +
            (event.metadata['agent']['position']['z'] - self.goal_position['z'])**2
        )
        
        if dist_to_goal > self.success_distance:
            return False

        # Check if the target object is visible
        visible_objects = event.metadata.get('objects', [])
        for obj in visible_objects:
            if obj['objectId'] == self.target_object['objectId'] and obj['visible']:
                return True
        return False

    def seed(self, seed):
        """
        Sets the random seed. AI2-THOR doesn't have a single seed function,
        but we can seed Python's random module which we use for choices.
        """
        random.seed(seed)

    def close(self):
        """
        Closes the AI2-THOR controller.
        """
        if self.controller:
            self.controller.stop()