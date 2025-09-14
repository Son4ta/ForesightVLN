import warnings
warnings.filterwarnings('ignore')
import math
import os
import re
import cv2
from PIL import Image
import skimage.morphology
from skimage.draw import line_aa, line
import numpy as np
import torch
from torchvision import transforms

from src.utils.fmm.fmm_planner_policy import FMMPlanner
import src.utils.fmm.pose_utils as pu
from src.utils.visualization.semantic_prediction import SemanticPredMaskRCNN
from src.utils.visualization.visualization import (
    # init_vis_image,
    draw_line,
    get_contour_points,
    line_list,
    add_text_list,
    draw_frontiers,  # 导入新的绘图函数
    draw_tsp_path,    # 导入新的绘图函数
)
from src.utils.visualization.visualization import init_vis_image_v2 as init_vis_image # 导入新的布局函数 但是不改变调用

from src.utils.visualization.save import save_video
from src.utils.llm import LLM

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch



class UniGoal_Agent():
    def __init__(self, args, envs):
        self.args = args
        self.envs = envs
        self.device = args.device

        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])


        self.sem_pred = SemanticPredMaskRCNN(args)
        self.llm = LLM(self.args.base_url, self.args.api_key, self.args.llm_model)

        self.selem = skimage.morphology.disk(3)

        self.rgbd = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.instance_imagegoal = None
        self.text_goal = None

        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='disk').eval().to(self.device)

        self.global_width = args.global_width
        self.global_height = args.global_height
        self.local_width = args.local_width
        self.local_height = args.local_height
        
        self.global_goal = None
        # define a temporal goal with a living time
        self.temp_goal = None
        self.last_temp_goal = None # avoid choose one goal twice
        self.forbidden_temp_goal = []
        self.flag = 0
        self.goal_instance_whwh = None
        # define untraversible area of the goal: 0 means area can be goals, 1 means cannot be
        self.goal_map_mask = np.ones((self.global_width, self.global_height))
        self.pred_box = []
        self.prompt_text2object = '"chair: 0, sofa: 1, plant: 2, bed: 3, toilet: 4, tv_monitor: 5" The above are the labels corresponding to each category. Which object is described in the following text? Only response the number of the label and not include other text.\nText: {text}'
        torch.set_grad_enabled(False)

        if args.visualize:
            self.vis_image_background = None
            self.rgb_vis = None
            self.vis_image_list = []

    def reset(self):
        args = self.args

        obs, info = self.envs.reset()

        if self.args.goal_type == 'ins-image':
            self.instance_imagegoal = self.envs.instance_imagegoal
        elif self.args.goal_type == 'text':
            self.text_goal = self.envs.text_goal
        idx = self.get_goal_cat_id()
        if idx is not None:
            self.envs.set_goal_cat_id(idx)

        rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2).transpose(2, 0, 1)
        self.raw_obs = rgbd[:3, :, :].transpose(1, 2, 0)
        self.raw_depth = rgbd[3:4, :, :]

        rgbd, seg_predictions = self.preprocess_obs(rgbd)
        self.rgbd = rgbd

        self.obs_shape = rgbd.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.global_goal = None
        self.temp_goal = None
        self.last_temp_goal = None
        self.forbidden_temp_goal = []
        self.goal_map_mask = np.ones(map_shape)
        self.goal_instance_whwh = None
        self.pred_box = []
        self.been_stuck = False
        self.stuck_goal = None
        self.frontier_vis = None

        if args.visualize:
            self.vis_image_background = init_vis_image(self.envs.goal_name, self.args)

        return obs, rgbd, info

    def local_feature_match_lightglue(self, re_key2=False):
        with torch.set_grad_enabled(False):
            ob = numpy_image_to_torch(self.raw_obs[:, :, :3]).to(self.device)
            gi = numpy_image_to_torch(self.instance_imagegoal).to(self.device)
            try:
                feats0, feats1, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                    )
                # indices with shape (K, 2)
                matches = matches01['matches']
                # in case that the matches collapse make a check
                b = torch.nonzero(matches[..., 0] < 2048, as_tuple=False)
                c = torch.index_select(matches[..., 0], dim=0, index=b.squeeze())
                points0 = feats0['keypoints'][c]
                if re_key2:
                    return (points0.numpy(), feats1['keypoints'][c].numpy())
                else:
                    return points0.numpy()  
            except:
                if re_key2:
                    # print(f'{self.env.rank}  {self.env.timestep}  h')
                    return (np.zeros((1, 2)), np.zeros((1, 2)))
                else:
                    # print(f'{self.env.rank}  {self.env.timestep}  h')
                    return np.zeros((1, 2))
                
    def compute_ins_dis_v1(self, depth, whwh, k=3):
        '''
        analyze the maxium depth points's pos
        make sure the object is within the range of 10m
        '''
        hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            bins=200,range=(0,2000))
        peak_indices = np.argsort(hist)[-k:]  # Get the indices of the top k peaks
        peak_values = hist[peak_indices] + hist[np.clip(peak_indices-1, 0, len(hist)-1)]  + \
            hist[np.clip(peak_indices+1, 0, len(hist)-1)]
        max_area_index = np.argmax(peak_values)  # Find the index of the peak with the largest area
        max_index = peak_indices[max_area_index]
        # max_index = np.argmax(hist)
        return bins[max_index]

    def compute_ins_goal_map(self, whwh, start, start_o):
        goal_mask = np.zeros_like(self.rgbd[3, :, :])
        goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1
        semantic_mask = (self.rgbd[4+self.envs.gt_goal_idx, :, :] > 0) & (goal_mask > 0)

        depth_h, depth_w = np.where(semantic_mask > 0)
        goal_dis = self.rgbd[3, :, :][depth_h, depth_w] / self.args.map_resolution

        goal_angle = -self.args.hfov / 2 * (depth_w - self.rgbd.shape[2]/2) \
        / (self.rgbd.shape[2]/2)
        goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
            start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
        goal_map = np.zeros((self.local_width, self.local_height))
        goal[0] = np.clip(goal[0], 0, 240-1).astype(int)
        goal[1] = np.clip(goal[1], 0, 240-1).astype(int)
        goal_map[goal[0], goal[1]] = 1
        return goal_map

    def instance_discriminator(self, planner_inputs, id_lo_whwh_speci):
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        goal_mask = self.rgbd[4+self.envs.gt_goal_idx, :, :]

        if self.instance_imagegoal is None and self.text_goal is None:
            # not initialized
            return planner_inputs
        elif self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
            return planner_inputs
        elif self.been_stuck:
            
            planner_inputs['found_goal'] = 0
            if self.stuck_goal is None:

                navigable_indices = np.argwhere(self.visited[gx1:gx2, gy1:gy2] > 0)
                goal = np.array([0, 0])
                for _ in range(100):
                    random_index = np.random.choice(len(navigable_indices))
                    goal = navigable_indices[random_index]
                    if pu.get_l2_distance(goal[0], start[0], goal[1], start[1]) > 16:
                        break

                goal = pu.threshold_poses(goal, map_pred.shape)                
                self.stuck_goal = [int(goal[0])+gx1, int(goal[1])+gy1]
            else:
                goal = np.array([self.stuck_goal[0]-gx1, self.stuck_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
            planner_inputs['goal'] = np.zeros((self.local_width, self.local_height))
            planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
        elif planner_inputs['found_goal'] == 1:
            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2] / 4).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            goal_mask = np.zeros_like(goal_mask)
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            if self.args.goal_type == 'ins-image':
                index = self.local_feature_match_lightglue()
                match_points = index.shape[0]
            planner_inputs['found_goal'] = 0

            if self.temp_goal is not None:
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
            else:
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)
                if not np.any(goal_map>0) :
                    tgoal_dis = self.compute_ins_dis_v1(self.rgbd[3, :, :], whwh) / self.args.map_resolution
                    rgb_center = np.array([whwh[3]+whwh[1], whwh[2]+whwh[0]])//2
                    goal_angle = -self.args.hfov / 2 * (rgb_center[1] - self.rgbd.shape[2]/2) \
                    / (self.rgbd.shape[2]/2)
                    goal = [start[0]+tgoal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                        start[1]+tgoal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
                    goal = pu.threshold_poses(goal, map_pred.shape)
                    rr,cc = skimage.draw.ellipse(goal[0], goal[1], 10, 10, shape=goal_map.shape)
                    goal_map[rr, cc] = 1


                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)

            if goal_dis is None:
                self.temp_goal = None
                planner_inputs['goal'] = planner_inputs['exp_goal']
                selem = skimage.morphology.disk(3)
                goal_map = skimage.morphology.dilation(goal_map, selem)
                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                print(f"Rank: {self.envs.rank}, timestep: {self.envs.timestep},  temp goal unavigable !")
            else:
                if self.args.goal_type == 'ins-image' and match_points > 100:
                    planner_inputs['found_goal'] = 1
                    global_goal = np.zeros((self.global_width, self.global_height))
                    global_goal[gx1:gx2, gy1:gy2] = goal_map
                    self.global_goal = global_goal
                    planner_inputs['goal'] = goal_map
                    self.temp_goal = None
                else:
                    if (self.args.goal_type == 'ins-image' and goal_dis < 50) or (self.args.goal_type == 'text' and goal_dis < 15):
                        if (self.args.goal_type == 'ins-image' and match_points > 90) or self.args.goal_type == 'text':
                            planner_inputs['found_goal'] = 1
                            global_goal = np.zeros((self.global_width, self.global_height))
                            global_goal[gx1:gx2, gy1:gy2] = goal_map
                            self.global_goal = global_goal
                            planner_inputs['goal'] = goal_map
                            self.temp_goal = None
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
                            selem = skimage.morphology.disk(1)
                            goal_map = skimage.morphology.dilation(goal_map, selem)
                            self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                    else:
                        new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                        if np.any(new_goal_map > 0):
                            planner_inputs['goal'] = new_goal_map
                            temp_goal = np.zeros((self.global_width, self.global_height))
                            temp_goal[gx1:gx2, gy1:gy2] = new_goal_map
                            self.temp_goal = temp_goal
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
            return planner_inputs

        else:
            planner_inputs['goal'] = planner_inputs['exp_goal']
            if self.temp_goal is not None:  
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
                planner_inputs['found_goal'] = 0
                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                if np.any(new_goal_map > 0):
                    if goal_dis is not None:
                        planner_inputs['goal'] = new_goal_map
                        if goal_dis < 100:
                            if self.args.goal_type == 'ins-image':
                                index = self.local_feature_match_lightglue()
                                match_points = index.shape[0]
                            if (self.args.goal_type == 'ins-image' and match_points < 80) or self.args.goal_type == 'text':
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                selem = skimage.morphology.disk(3)
                                new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                                self.temp_goal = None
                    else:
                        selem = skimage.morphology.disk(3)
                        new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                        self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                        self.temp_goal = None
                        print(f"Rank: {self.envs.rank}, timestep: {self.envs.timestep},  temp goal unavigable !")
                else:
                    self.temp_goal = None
                    
                    
            return planner_inputs


    def step(self, agent_input):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if agent_input["wait"]:
            self.last_action = None
            self.envs.info["sensor_pose"] = [0., 0., 0.]
            return None, np.zeros(self.rgbd.shape), False, self.envs.info

        id_lo_whwh = self.pred_box


        id_lo_whwh_speci = [id_lo_whwh[i] for i in range(len(id_lo_whwh)) \
                    if id_lo_whwh[i][0] == self.envs.gt_goal_idx]


        agent_input["found_goal"] = (id_lo_whwh_speci != [])

        self.instance_discriminator(agent_input, id_lo_whwh_speci)

        action = self.get_action(agent_input)

        if self.args.visualize:
            self.visualize(agent_input)

        if action >= 0:
            action = {'action': action}
            obs, done, info = self.envs.step(action)
                        
            rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2).transpose(2, 0, 1)
            self.raw_obs = rgbd[:3, :, :].transpose(1, 2, 0)
            self.raw_depth = rgbd[3:4, :, :]

            rgbd, seg_predictions = self.preprocess_obs(rgbd) 
            self.last_action = action['action']
            self.rgbd = rgbd

            # if done:
            #     obs, rgbd, info = self.reset()

            return obs, rgbd, done, self.envs.info

        else:
            self.last_action = None
            self.envs.info["sensor_pose"] = [0., 0., 0.]
            return None, np.zeros(self.obs_shape), False, self.envs.info

    def get_action(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)\
        
        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        # self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
        #                                start[1] - 0:start[1] + 1] = 1
        rr, cc, _ = line_aa(last_start[0], last_start[1], start[0], start[1])
        self.visited[gx1:gx2, gy1:gy2][rr, cc] += 1

        if args.visualize:            
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # relieve the stuck goal
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        if abs(x1 - x2) >= 0.05 or abs(y1 - y2) >= 0.05:
            self.been_stuck = False
            self.stuck_goal = None

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                    self.been_stuck = True
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1                

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        local_goal, stop = self.get_local_goal(map_pred, start, np.copy(goal),
                                  planning_window)

        if stop and planner_inputs['found_goal'] == 1:
            action = 0
        else:
            (local_x, local_y) = local_goal
            angle_st_goal = math.degrees(math.atan2(local_x - start[0],
                                                    local_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle:
                action = 3
            elif relative_angle < -self.args.turn_angle:
                action = 2
            else:
                action = 1

        return action

    def get_local_goal(self, grid, start, goal, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = self.add_boundary(traversible)
        goal = self.add_boundary(goal, value=0)
        visited = self.add_boundary(self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2], value=0)

        planner = FMMPlanner(traversible)
        if self.global_goal is not None or self.temp_goal is not None:
            selem = skimage.morphology.disk(10)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        elif self.stuck_goal is not None:
            selem = skimage.morphology.disk(1)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        else:
            selem = skimage.morphology.disk(3)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]


        if self.global_goal is not None:
            st_dis = pu.get_l2_dis_point_map(state, goal) * self.args.map_resolution
            fmm_dist = planner.fmm_dist * self.args.map_resolution 
            dis = fmm_dist[start[0]+1, start[1]+1]
            if st_dis < 100 and dis/st_dis > 2:
                return (0, 0), True

        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        if replan:
            stg_x, stg_y, _, stop = planner.get_short_term_goal(state, 2)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1:h + 1, 1:w + 1] = mat
        return new_mat

    def compute_temp_goal_distance(self, grid, goal_map, start, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape
        goal = goal_map * 1
        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        st_dis = pu.get_l2_dis_point_map(start, goal) * self.args.map_resolution  # cm

        traversible = self.add_boundary(traversible)
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        
        goal = cv2.dilate(goal, selem)
        
        goal = self.add_boundary(goal, value=0)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution 
        dis = fmm_dist[start[0]+1, start[1]+1]

        return dis
        if dis < fmm_dist.max() and dis/st_dis < 2:
            return dis
        else:
            return None

    def preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        sem_seg_pred, seg_predictions = self.pred_sem(
            rgb.astype(np.uint8), use_seg=use_seg)

        if args.environment == 'habitat':
            depth = self.preprocess_depth(depth, args.min_depth, args.max_depth)
        elif self.args.environment == 'ai2thor':
            # 如果是 AI2-THOR 环境，深度单位是米。
            # BEV 建图模块期望的单位是厘米（cm）。
            # 因此，我们只需要将深度值乘以 100 即可。
            # depth = self.preprocess_depth(depth, args.min_depth, args.max_depth)
            # depth = depth * 100.0
            pass
        

        '''
        作用: 计算整数“降采样因子” (ds)。
        args.env_frame_width: 这是模拟器（如 AI2-THOR 或 Habitat）输出的原始图像宽度，例如 640 像素。
        args.frame_width: 这是 Agent 的神经网络模型期望接收的图像宽度，通常会更小以便于计算，例如 160 像素。
        '''
        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            # rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            # 对 RGB 图像使用与 Depth 和 Semantic 完全相同的跳点采样方法
            rgb = rgb[ds // 2::ds, ds // 2::ds]
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]
        
        if self.args.save_debug_images:
            import src.utils.debug as debug
            debug.save_debug_images(
                rgb_image=rgb,
                depth_image=depth,
                timestep=self.envs.timestep,
                episode_id=self.envs.episode_no, # 从环境中获取 episode 编号
                save_dir=self.args.debug_save_dir # 从配置中获取保存路径
            )

        # 这里要是AI2-THOR环境的话，depth不需要再包装一层 
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=2)

        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state, seg_predictions

    def preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[0]):
            depth[i, :][depth[i, :] == 0.] = depth[i, :].max() + 0.01

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def pred_sem(self, rgb, depth=None, use_seg=True, pred_bbox=False):
        if pred_bbox:
            semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
            return self.pred_box, seg_predictions
        else:
            if use_seg:
                semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
                semantic_pred = semantic_pred.astype(np.float32)
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
            return semantic_pred, seg_predictions
        
    def get_goal_cat_id(self):
        if self.args.goal_type == 'ins-image':
            instance_whwh, seg_predictions = self.pred_sem(self.instance_imagegoal.astype(np.uint8), None, pred_bbox=True)
            ins_whwh = [instance_whwh[i] for i in range(len(instance_whwh)) \
                if (instance_whwh[i][2][3]-instance_whwh[i][2][1])>1/6*self.instance_imagegoal.shape[0] or \
                    (instance_whwh[i][2][2]-instance_whwh[i][2][0])>1/6*self.instance_imagegoal.shape[1]]
            if ins_whwh != []:
                ins_whwh = sorted(ins_whwh,  \
                    key=lambda s: ((s[2][0]+s[2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((s[2][1]+s[2][3]-self.instance_imagegoal.shape[0])/2)**2 \
                    )
                if ((ins_whwh[0][2][0]+ins_whwh[0][2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((ins_whwh[0][2][1]+ins_whwh[0][2][3]-self.instance_imagegoal.shape[0])/2)**2 < \
                            ((self.instance_imagegoal.shape[1] / 6)**2 )*2:
                    return int(ins_whwh[0][0])
            return None
        elif self.args.goal_type == 'text':
            for i in range(10):
                if isinstance(self.text_goal, dict) and 'intrinsic_attributes' in self.text_goal:  
                    text_goal = self.text_goal['intrinsic_attributes']
                else:
                    text_goal = self.text_goal
                text_goal_id = self.llm(self.prompt_text2object.replace('{text}', text_goal))
                try:
                    text_goal_id = re.findall(r'\d+', text_goal_id)[0]
                    text_goal_id = int(text_goal_id)
                    if 0 <= text_goal_id < 6:
                        return text_goal_id
                except:
                    pass
            return 0

    # This is the new helper function that encapsulates your original map drawing logic.
    def _create_categorical_map(self, inputs):
        """
        >>> v3.2: 修复渲染顺序 (BEV在最底层) <<<
        """
        args = self.args
        # This is the beautiful color palette you provided
        color_palette = [
            1.0, 1.0, 1.0,       # 0: unexplored
            0.6, 0.6, 0.6,       # 1: obstacles
            0.95, 0.95, 0.95,    # 2: explored
            0.96, 0.36, 0.26,    # 3: visited
            0.12, 0.47, 0.70,    # 4: goal
            # Semantic classes start here
            0.94, 0.78, 0.66, 0.88, 0.94, 0.66, 0.66, 0.94, 0.85,
            0.71, 0.66, 0.94, 0.92, 0.66, 0.94, 0.94, 0.66, 0.74,
        ] * 3

        map_pred = np.rint(inputs['map_pred'])
        exp_pred = np.rint(inputs['exp_pred'])
        sem_map_cat = inputs['sem_map_pred'].copy()
        gx1, gx2, gy1, gy2 = map(int, inputs['pose_pred'][3:])
        map_pred[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 1

        map_h, map_w = map_pred.shape
        # --- NEW Layered Rendering Logic ---
        # L0: Start with a blank map (unexplored)
        final_map = np.zeros((map_h, map_w), dtype=np.uint8)
        
        # L1: Draw explored area
        final_map[exp_pred == 1] = 2
        
        # L2: Draw semantic categories on top of explored area
        semantic_mask = (sem_map_cat > 0) & (exp_pred == 1)
        final_map[semantic_mask] = sem_map_cat[semantic_mask] + 5

        # L3: Draw obstacles on top of everything except goal and visited
        final_map[map_pred == 1] = 1
        
        # L4: Draw visited path, this has high priority
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        final_map[vis_mask] = 3

        # L5: Draw goal, this has the highest priority
        goal_mat = inputs['goal']
        if goal_mat.any():
            selem = skimage.morphology.disk(4)
            goal_mask = skimage.morphology.binary_dilation(goal_mat, selem)
            final_map[goal_mask] = 4
        # --- End of Layered Rendering ---

        color_pal_int = [int(x * 255.) for x in color_palette]
        sem_map_vis_pil = Image.new("P", (map_w, map_h))
        sem_map_vis_pil.putpalette(color_pal_int)
        sem_map_vis_pil.putdata(final_map.flatten().astype(np.uint8))
        sem_map_vis_pil = sem_map_vis_pil.convert("RGB")
        
        sem_map_vis_np = np.array(sem_map_vis_pil)
        sem_map_vis_np = cv2.cvtColor(sem_map_vis_np, cv2.COLOR_RGB2BGR)
        
        return sem_map_vis_np.copy()


    def visualize(self, inputs):
        """
        >>> v3.5: 修正了坐标轴混淆和flipud变换问题，实现箭头位置和姿态的精确匹配 <<<
        """
        args = self.args
        vis_image = self.vis_image_background.copy()
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        margin, width, height = 30, 1800, 900
        col1_width = 800
        col2_width = width - col1_width - 3 * margin
        col2_x_start = col1_width + 2 * margin

        # === Panel 1: Agent's Egocentric View ===
        p1_x, p1_y, p1_w, p1_h = margin, margin, col1_width, height - 2 * margin
        rgb_resized = cv2.resize(self.rgb_vis, (p1_w, p1_h), interpolation=cv2.INTER_AREA)
        vis_image[p1_y : p1_y + p1_h, p1_x : p1_x + p1_w] = rgb_resized

        # === Panel 2: Global Map & Planning ===
        p2_h = int((height - 3 * margin) * 0.6)
        p2_x, p2_y, p2_w = col2_x_start, margin, col2_width

        # 1. 创建具有正确渲染层次的基础地图
        planning_map = self._create_categorical_map(inputs)
        map_h, map_w, _ = planning_map.shape

        # 2. 翻转地图以实现 "Y轴向上" 的效果
        final_canvas = np.flipud(planning_map).copy()

        # 3. 在新的“Y轴向上”的画布上计算和绘制所有元素

        # 3.1 绘制智能体箭头 (漂移和90度偏差问题修复)
        # --- FIX START ---

        # 3.1.1: 将全局米制坐标转换为全局像素坐标 (保持x,y对应关系)
        agent_global_px_x = start_x * 100. / args.map_resolution
        agent_global_px_y = start_y * 100. / args.map_resolution
        
        # 3.1.2: 转换为局部地图坐标 (未翻转前)
        # 修正坐标轴对应：x对应gy(列), y对应gx(行)
        agent_local_px_x = agent_global_px_x - gy1
        agent_local_px_y = agent_global_px_y - gx1

        # 3.1.3: 将局部坐标转换为翻转后画布(final_canvas)的坐标
        # x坐标不变
        draw_x = agent_local_px_x
        # y坐标需要根据flipud进行变换
        draw_y = (map_h - 1) - agent_local_px_y 

        # 3.1.4: 角度转换 (您的原始逻辑是正确的，无需修改)
        # agent_o (顺时针, 0朝上) -> math angle (逆时针, 0朝右)
        angle_rad = np.deg2rad(-start_o)

        # 使用修正后的坐标和角度绘制箭头
        agent_arrow = get_contour_points(pos=(draw_x, draw_y, angle_rad), origin=(0, 0), size=12)
        cv2.drawContours(final_canvas, [agent_arrow], 0, (0, 0, 255), -1) # 红色箭头

        # --- FIX END ---

        # 3.2 绘制前沿点和TSP路径 (同样需要修正Y轴)
        if 'all_frontiers' in inputs and inputs['all_frontiers'] is not None:
            frontiers_local = inputs['all_frontiers'].copy().astype(np.float32)
            # 修正坐标转换
            frontiers_draw_x = frontiers_local[:, 1] - gy1 
            frontiers_draw_y = (map_h - 1) - (frontiers_local[:, 0] - gx1)
            frontiers_to_draw = np.vstack((frontiers_draw_x, frontiers_draw_y)).T
            draw_frontiers(final_canvas, frontiers_to_draw.astype(np.int32), color=(0, 165, 255), radius=4)

        if 'tsp_path_info' in inputs and inputs['tsp_path_info'] is not None:
            tsp_info = inputs['tsp_path_info']
            tsp_frontiers_local = tsp_info['frontiers'].copy().astype(np.float32)
            # 修正坐标转换
            tsp_frontiers_draw_x = tsp_frontiers_local[:, 1] - gy1
            tsp_frontiers_draw_y = (map_h - 1) - (tsp_frontiers_local[:, 0] - gx1)
            tsp_frontiers_to_draw = np.vstack((tsp_frontiers_draw_x, tsp_frontiers_draw_y)).T
            draw_frontiers(final_canvas, tsp_frontiers_to_draw.astype(np.int32), color=(255, 255, 0), radius=6)

            if 'full_path_coords' in tsp_info:
                path_local = tsp_info['full_path_coords'].copy().astype(np.float32)
                # 修正坐标转换
                path_draw_x = path_local[:, 1] - gy1
                path_draw_y = (map_h - 1) - (path_local[:, 0] - gx1)
                path_to_draw = np.vstack((path_draw_x, path_draw_y)).T
                draw_tsp_path(final_canvas, path_to_draw.astype(np.int32), np.arange(len(path_local)), color=(255, 0, 255), thickness=3)

        # 4. 直接缩放和粘贴最终的画布
        planning_map_resized = cv2.resize(final_canvas, (p2_w, p2_h), interpolation=cv2.INTER_NEAREST)
        vis_image[p2_y : p2_y + p2_h, p2_x : p2_x + p2_w] = planning_map_resized
        
        # === Panel 3: 任务目标 ===
        p3_y_start = p2_y + p2_h + margin
        p3_h = height - p3_y_start - margin
        p3_w = int((col2_width - margin) * 0.35)
        p3_x = col2_x_start

        goal_panel = np.ones((p3_h, p3_w, 3), dtype=np.uint8) * 255

        if args.goal_type == 'ins-image' and self.instance_imagegoal is not None:
            goal_img = self.instance_imagegoal
            h, w, _ = goal_img.shape

            if h > w:
                goal_img = goal_img[h // 2 - w // 2 : h // 2 + w // 2, :]
            elif w > h:
                goal_img = goal_img[:, w // 2 - h // 2 : w // 2 + h // 2]

            goal_h, goal_w, _ = goal_img.shape
            scale = min(p3_w / goal_w, p3_h / goal_h) * 0.9
            new_w, new_h = int(goal_w * scale), int(goal_h * scale)

            goal_resized = cv2.resize(goal_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            start_x_panel = (p3_w - new_w) // 2
            start_y_panel = (p3_h - new_h) // 2

            goal_panel[start_y_panel:start_y_panel+new_h, start_x_panel:start_x_panel+new_w] = \
                cv2.cvtColor(goal_resized, cv2.COLOR_RGB2BGR)

        elif args.goal_type == 'text' and self.text_goal is not None:
            if isinstance(self.text_goal, dict) and 'intrinsic_attributes' in self.text_goal:
                text_goal = self.text_goal['intrinsic_attributes'] + ' ' + self.text_goal.get('extrinsic_attributes', '')
            else:
                text_goal = str(self.text_goal)

            text_list_proc = line_list(text_goal, line_length=20)[:8]
            add_text_list(goal_panel, text_list_proc, font_scale=0.6, thickness=1, color=(20, 20, 20))

        vis_image[p3_y_start : p3_y_start + p3_h, p3_x : p3_x + p3_w] = goal_panel

        # === Panel 4: 语义价值图 ===
        p4_x_start = p3_x + p3_w + margin
        p4_w = col2_width - p3_w - margin
        p4_x, p4_y = p4_x_start, p3_y_start

        if 'semantic_map_vis' in inputs:
            semantic_map_resized = cv2.resize(inputs['semantic_map_vis'], (p4_w, p3_h), interpolation=cv2.INTER_NEAREST)
            vis_image[p4_y : p4_y + p3_h, p4_x : p4_x + p4_w] = semantic_map_resized

        self.vis_image_list.append(vis_image)

        # 保存图像
        tmp_dir = 'outputs/tmp'
        os.makedirs(tmp_dir, exist_ok=True)

        height, width, layers = vis_image.shape
        image_name = 'debug.jpg' if self.args.is_debugging else 'v.jpg'
        cv2.imwrite(os.path.join(tmp_dir, image_name), cv2.resize(vis_image, (width // 2, height // 2)))


        self.vis_image_list.append(vis_image)

        tmp_dir = 'outputs/tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        height, width, layers = vis_image.shape
        if self.args.is_debugging:
            image_name = 'debug.jpg'
        else:
            image_name = 'v.jpg'
        cv2.imwrite(os.path.join(tmp_dir, image_name), cv2.resize(vis_image, (width // 2, height // 2)))

    
    def save_visualization(self, video_path):
        save_video(self.vis_image_list, video_path, fps=15, input_color_space="BGR")
        self.vis_image_list = []
