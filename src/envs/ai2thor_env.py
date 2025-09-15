# src/envs/ai2thor_env.py

import ai2thor.controller
from ai2thor.platform import CloudRendering
from types import SimpleNamespace
import numpy as np
import quaternion
import random
import math
from src.utils.fmm.pose_utils import get_rel_pose_change

# Try to import category mappings from the project; use a default if it fails
try:
    from configs.categories import name2index, hm3d_to_coco
    # You may need to create a new mapping for ProcTHOR
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
            # Output width and height
            width=self.args.env_frame_width,
            height=self.args.env_frame_height,
            fieldOfView=self.args.hfov,
            horizon=0.0, # Look straight ahead
            # ↓ Add headless mode for server-side running
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

        # --- For compatibility with upper-level calls, update goal_object_id and object_category ---
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
        # Directly get the correct action name from the mapping ("MoveAhead", "RotateLeft", etc.)
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
            According to the official AI2-THOR documentation, RotateLeft and RotateRight rotate by a fixed angle by default.
            To rotate by a custom angle, use the Rotate action and specify the degrees parameter.
            Positive is right, negative is left.
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
        
        # Convert AI2-THOR's left-handed coordinate position (x, y, z) to Habitat's right-handed coordinate (x, y, -z)
        # This step is critical for point cloud projection
        habitat_position = np.array([pos['x'], pos['y'], -pos['z']], dtype=np.float32)

        # The yaw angle definition of AI2-THOR (left-handed, y-up) and Habitat (right-handed, y-up) is consistent
        yaw_rad = np.deg2rad(-rot_deg)
        q = quaternion.from_euler_angles(0, yaw_rad, 0)

        return {'position': habitat_position, 'rotation': q}

    def _get_agent_location(self):
        # This function defines the project's "map coordinate system"
        # It generates a 2D map coordinate (map_x, map_y) and orientation from 3D world coordinates
        agent_pose_hab = self.get_agent_pose()
        # pos_hab is (x, y, z) in Habitat coordinates
        pos_hab = agent_pose_hab['position']
        
        # Convert 3D coordinates to 2D map coordinates
        # x_map = -z_hab, y_map = -x_hab
        map_x = -pos_hab[2]
        map_y = -pos_hab[0]
        
        yaw_rad = quaternion.as_euler_angles(agent_pose_hab['rotation'])[1]
        
        return map_x, map_y, yaw_rad

    def _get_location_change(self):
        # ===============================================
        #  >>> Odometer calculation <<<
        #  Calculate relative displacement and rotation change, x+ is forward, r+ is left turn, r- is right turn
        #  Note: AI2-THOR's coordinate system is left-handed
        # ===============================================
        dx, dy, do = 0.0, 0.0, 0.0
        current_location = self._get_agent_location()
        action_str = self.last_agent_action
        if action_str == None:
            # If there is no action, return zero change
            return [0.0, 0.0, 0.0]
        elif action_str == "MoveAhead":
            # Get the pose in map coordinates for the previous and current frames
            x1, y1, o1 = self.last_agent_location
            x2, y2, o2 = current_location
            dx = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # Moving forward means moving a fixed step length forward
            # dx = self.args.grid_size
        elif action_str == "RotateLeft":
            # Left turn means rotating counterclockwise by a fixed angle
            do = np.deg2rad(self.rotation_degrees)
        elif action_str == "RotateRight":
            # Right turn means rotating clockwise by a fixed angle
            do = -np.deg2rad(self.rotation_degrees)
        
        # [forward displacement, lateral displacement, counterclockwise rotation]
        self.last_agent_location = current_location
        return dx, dy, do
        
        current_location = self._get_agent_location()
        x1, y1, o1 = self.last_agent_location
        x2, y2, o2 = current_location
        
        dx, dy, do = get_rel_pose_change(
            current_location, self.last_agent_location)
        self.last_agent_location = current_location

        # Calculate displacement in world coordinates
        dx_world = x2 - x1
        dy_world = y2 - y1

        # Rotate the world displacement into the previous local coordinate system
        # sin(-o1) = -sin(o1), cos(-o1) = cos(o1)
        dx_local = dx_world * np.cos(o1) + dy_world * np.sin(o1)
        dy_local = -dx_world * np.sin(o1) + dy_world * np.cos(o1)
        
        # Calculate orientation change
        do = o2 - o1
        # Normalize the angle change to [-pi, pi]
        if do > np.pi:
            do -= 2 * np.pi
        if do < -np.pi:
            do += 2 * np.pi

        self.last_agent_location = current_location

        # The mapping module of UniGoal expects sensor_pose as [forward movement distance, lateral movement distance, counterclockwise rotation angle]
        # dx_local is forward, dy_local is right, do is counterclockwise rotation
        # Note: dx_local is in meters, and the BEV module's pose is also in meters. Units are consistent.
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
        # Create an empty SimpleNamespace object, because we can't directly change self.current_episode
        fake_episode = SimpleNamespace()
        # TODO There is a big problem here, asset_id = 'Wall_Decor_Photo_6', the logic below needs to be rewritten
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

        # TODO ☠ Changing self.current_episode here is expected to have no effect, as the subsequent process does not use current_episode. Please verify.
        self.current_episode = fake_episode



    def _get_goal_image(self):
        """
        In the scene, find an object instance from a predefined target list and generate a clear image of it
        while maintaining a minimum safe distance. This process does not move the main Agent.
        """
        # 1. Define a list of target object categories
        # The function will randomly look for an object of one of the following types
        TARGET_OBJECT_TYPES = [
            "Chair", "Sofa", "ArmChair", "Bed", "Desk"
        ]

        # 2. Find any instance of the listed types in the scene
        all_objects = self.controller.last_event.metadata['objects']
        
        # Filter all object instances belonging to the target list
        possible_targets = [obj for obj in all_objects if obj['objectType'] in TARGET_OBJECT_TYPES]
        
        if not possible_targets:
            print(f"Error: No objects from the list {TARGET_OBJECT_TYPES} found in the scene.")
            return np.zeros((self.args.frame_height, self.args.frame_width, 3), dtype=np.uint8)

        # Randomly select one from the found instances
        target_object_info = random.choice(possible_targets)
        goal_position = target_object_info['position']
        print(f"Selected target instance: {target_object_info['objectId']} (type: {target_object_info['objectType']})")

        # 3. Find the best camera viewpoint that meets the minimum distance requirement
        try:
            reachable_pos = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
            if not reachable_pos:
                raise ValueError("No reachable positions found in the environment.")
        except Exception as e:
            print(f"Error getting reachable positions: {e}")
            return np.zeros((self.args.frame_height, self.args.frame_width, 3), dtype=np.uint8)

        # ===================================================================
        # >>> Core modification: Filter all reachable points at distance >= 0.5m from the target <<<
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
            print(f"Warning: No reachable viewpoints at least {MIN_DISTANCE}m from the target found.")
            return np.zeros((self.args.frame_height, self.args.frame_width, 3), dtype=np.uint8)

        # From the valid viewpoints, find the one closest to the target
        best_viewpoint_info = min(valid_viewpoints, key=lambda x: x['dist'])
        best_viewpoint = best_viewpoint_info['pos']
        print(f"Found best viewpoint, distance to target {best_viewpoint_info['dist']:.2f}m")

        # 4. Calculate camera position and rotation to face the target
        camera_position = {'x': best_viewpoint['x'], 'y': best_viewpoint['y'] + 0.6, 'z': best_viewpoint['z']}
        direction_vector = {'x': goal_position['x'] - camera_position['x'], 'z': goal_position['z'] - camera_position['z']}
        yaw_angle = math.degrees(math.atan2(direction_vector['x'], direction_vector['z']))
        pitch_angle = 30.0
        camera_rotation = {'x': pitch_angle, 'y': yaw_angle, 'z': 0}

        # 5. Add a third-person camera and capture the image
        event = self.controller.step(
            action="AddThirdPartyCamera",
            position=camera_position,
            rotation=camera_rotation,
            fieldOfView=105
        )

        if not event.third_party_camera_frames or len(event.third_party_camera_frames) == 0:
            print("Error: No image obtained after adding third-party camera.")
            return np.zeros((self.args.frame_height, self.args.frame_width, 3), dtype=np.uint8)

        # 6. Process image format and return
        rgba_frame = event.third_party_camera_frames[0]
        rgb_frame = rgba_frame[:, :, :3].astype(np.uint8)

        print(f"Successfully generated goal image for {target_object_info['objectId']}.")
        return rgb_frame

    def _format_observation(self, event):
        # ===============================================
        #  >>> Core fix 3: Simplify and correct sensor simulation <<<
        #  Ensure depth map and coordinate system match the mapping module's expectations
        # ===============================================
        obs = {}
        obs['rgb'] = event.frame.astype(np.uint8)
        
        # 1. Depth map processing: filter invalid values and convert to centimeters
        depth_meters = event.depth_frame.astype(np.float32)
        depth_meters[depth_meters >= self.args.max_depth] = 0 # Filter distant points
        depth_meters[depth_meters <= self.args.min_depth] = 0 # Filter close points
        obs['depth'] = np.expand_dims(depth_meters * 100.0, axis=2) # Convert to centimeters

        # 2. GPS and Compass
        # get_agent_pose() has already handled coordinate transformation
        agent_pose_hab = self.get_agent_pose()
        obs['gps'] = agent_pose_hab['position'][[2, 0]] # [-z, -x] -> [-z, x] if needed. Let's use [-z, x]
        obs['gps'][1] *= -1.0
        
        yaw_rad = quaternion.as_euler_angles(agent_pose_hab['rotation'])[1]
        obs['compass'] = np.array([yaw_rad], dtype=np.float32)

        # ===============================================
        #  >>> Add instance_imagegoal here <<<
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