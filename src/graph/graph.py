import os
import sys
import cv2
import numpy as np
import torch
import math
import dataclasses
import omegaconf
from openai import OpenAI
import base64
from io import BytesIO
import supervision as sv
import random
from PIL import Image
from sklearn.cluster import DBSCAN  
from collections import Counter 
from omegaconf import DictConfig
from pathlib import PosixPath, Path
from supervision.draw.color import ColorPalette
import copy
import skimage

from .graphbuilder import GraphBuilder
from .goalgraphdecomposer import GoalGraphDecomposer
from .overlap import GraphMatcher
from .scenegraphcorrector import SceneGraphCorrector
from .utils.slam_classes import MapObjectList
from .utils.utils import filter_objects, gobs_to_detection_list
from .utils.mapping import compute_spatial_similarities, merge_detections_to_objects

from ..utils.fmm.fmm_planner import FMMPlanner
from ..utils.fmm import pose_utils as pu
from ..utils.camera import get_camera_matrix
from ..utils.map import remove_small_frontiers
from ..utils.llm import LLM, VLM

sys.path.append('third_party/Grounded-Segment-Anything/')
from grounded_sam_demo import load_model, get_grounding_output
import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor
from lightglue import LightGlue, DISK
from lightglue.utils import match_pair , numpy_image_to_torch
# Import python-tsp library for solving Traveling Salesman Problem
from python_tsp.exact import solve_tsp_dynamic_programming

from collections import deque # <<< Efficiently implement fixed-length queue
from scipy.spatial.distance import cdist # <<< Fast distance calculation
from sklearn.cluster import DBSCAN # <<< Clustering

import json
import re

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []


class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        self.corr_score = 0
        self.center = None
        self.center_node = None
        self.nodes = []
        self.edges = set()
    
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def get_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."

class ObjectNode():
    def __init__(self):
        self.is_new_node = True
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None

    def set_relation(self, relation):
        self.relation = relation

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text


class SubGraph():
    def __init__(self, center_node):
        self.center_node = center_node
        self.edges = self.center_node.edges
        self.center = self.center_node.center
        self.nodes = set()
        for edge in self.edges:
            self.nodes.add(edge.node1)
            self.nodes.add(edge.node2)

    def get_subgraph_2_text(self):
        text = ''
        edges = set()
        for node in self.nodes:
            text = text + node.caption + '/'
            edges.update(node.edges)
        text = text[:-1] + '\n'
        for edge in edges:
            text = text + edge.relation + '/'
        text = text[:-1]
        return text


class Graph():
    def __init__(self, args, is_navigation=True) -> None:
        self.args = args
        self.map_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm
        self.map_size = args.map_size
        self.camera_matrix = get_camera_matrix(args.env_frame_height, args.env_frame_width, args.hfov)
        full_width, full_height = self.map_size, self.map_size
        self.full_width = full_width
        self.full_height = full_height
        self.visited = torch.zeros(full_width, full_height).float().cpu().numpy()
        self.device = args.device
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        self.is_navigation = is_navigation
        self.set_cfg()
        
        self.groundingdino_config_file = 'third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.sam_version = 'vit_h'
        self.sam_checkpoint = 'data/models/sam_vit_h_4b8939.pth'
        self.segment2d_results = []
        self.max_detections_per_object = 10
        self.threshold_list = {'bathtub': 3, 'bed': 3, 'cabinet': 2, 'chair': 1, 'chest_of_drawers': 3, 'clothes': 2, 'counter': 1, 'cushion': 3, 'fireplace': 3, 'gym_equipment': 2, 'picture': 3, 'plant': 3, 'seating': 0, 'shower': 2, 'sink': 2, 'sofa': 2, 'stool': 2, 'table': 1, 'toilet': 3, 'towel': 2, 'tv_monitor': 0}
        self.found_goal_times_threshold = 1
        self.N_max = 10
        self.node_space = 'table. tv. chair. cabinet. sofa. bed. windows. kitchen. bedroom. living room. mirror. plant. curtain. painting. picture'
        self.relations = ["next to", "opposite to", "below", "behind", "in front of"]
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character.
Examples:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to
Object pair(s):
        '''
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Only answer the room.'
        self.prompt_graph_corr_0 = 'What is the probability of A and B appearing together. [A:{}], [B:{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_1 = 'What else do you need to know to determine the probability of A and B appearing together? [A:{}], [B:{}]. Please output a short question (output only one sentence with no additional text).'
        self.prompt_graph_corr_2 = 'Here is the objects and relationships near A: [{}] You answer the following question with a short sentence based on this information. Question: {}'
        self.prompt_graph_corr_3 = 'The probability of A and B appearing together is about {}. Based on the dialog: [{}], re-determine the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_image2text = 'Describe the object at the center of the image and indicate the spatial relationship between other objects and it.'
        self.prompt_create_relation = """
Given the image, please analyze the spatial relationship between {obj1} and {obj2}.
If there is a clear spatial relationship, describe it using the following template:
"{obj1} is <relation type> {obj2}"
If no clear spatial relationship exists, state: "No clear spatial relationship between {obj1} and {obj2}"

Example output format for a relation:
"table and book: table is under book"

Example output format if no relation:
"No clear spatial relationship between table and book"

Please provide the relationship you can determine from the image.
        """
        self.grounded_sam = self.get_grounded_sam(self.device)
        self.llm = LLM(self.args.base_url, self.args.api_key, self.args.llm_model)
        self.vlm = VLM(self.args.base_url, self.args.api_key, self.args.vlm_model)
        self.graphbuilder = GraphBuilder(self.llm)
        self.goalgraphdecomposer = GoalGraphDecomposer(self.llm)
        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.image_matcher = LightGlue(features='disk').eval().to(self.device)

        # >>> New: Add core attributes for adaptive exploration strategy and TSP <<<
        # Core idea: Follow ApexNav, create and maintain a global semantic value map to guide exploration decisions.
        
        # 1. Semantic Score Map: Stores semantic relevance scores between each map grid and current target.
        #    Initialized to 0, same size as main map `full_map`.
        self.semantic_score_map = None # Will be initialized in set_full_map

        # 2. Semantic Confidence Map: Used for weighted average updates of semantic scores, enabling multi-frame information fusion.
        self.semantic_confidence_map = None # Will be initialized in set_full_map

        # 3. Frontier Locations: Stores coordinates of all currently detected frontier points [(y1, x1), (y2, x2), ...].
        #    This attribute will be updated in the get_goal function.
        self.frontier_locations = None
        
        # 4. Current Subgraph Goal: Used to share the current target to be found between different exploration strategies.
        self.subgraph = None
        
        # 5. Traversible Map: FMM Planner requires a binary map of traversible areas.
        self.traversible = None
        
        # 6. Agent Start Pose on Map: Starting coordinates of agent in traversible map.
        self.start = None

        # Add an attribute to store the previous free space map for calculating newly visible areas
        self.prev_free_map = None
        # Path recording
        self.tsp_path_info = None
        # >>> End of new attributes <<<

        # >>> New: Path history recording for optimizing exploration strategy <<<
        self.path_history = deque(maxlen=30) # Store the most recent 30 position points

                # >>> New: VLM context management module <<<
        # Long-term summary, updated regularly by LLM
        self.vlm_context_summary = "Mission has just started. I need to explore the environment to find the target." 
        # Short-term memory, stores recent N interactions
        self.vlm_short_term_memory = deque(maxlen=5) 
        # VLM judgment and confidence for finding target
        self.vlm_found_goal = False
        self.vlm_goal_confidence = 0.0
        # Counter for controlling summary update frequency
        self.vlm_update_counter = 0

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95,
               'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg

    def set_agent(self, agent):
        self.agent = agent

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_obj_goal(self, obj_goal):
        self.obj_goal = obj_goal

    def set_image_goal(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.instance_imagegoal = image
        text_goal = self.vlm(self.prompt_image2text, self.instance_imagegoal)
        self.set_text_goal(text_goal)

    def set_text_goal(self, text_goal):
        if isinstance(text_goal, dict) and 'intrinsic_attributes' in text_goal and 'extrinsic_attributes' in text_goal:
            text_goal = text_goal['intrinsic_attributes'] + ' ' + text_goal['extrinsic_attributes']
        self.text_goal = text_goal
        self.goalgraph = self.graphbuilder.build_graph_from_text(text_goal)
        self.goalgraph_decomposed = self.goalgraphdecomposer.goal_decomposition(self.goalgraph)

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map
        # >>> Modified: Initialize semantic maps here, ensuring they are created after `full_map` is created <<<
        if self.semantic_score_map is None:
            # Initialize a map of the same size as full_map to store VLM semantic scores
            self.semantic_score_map = torch.zeros_like(self.full_map[0,0], device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # Store confidence for each grid cell for weighted averaging
            self.semantic_confidence_map = torch.zeros_like(self.full_map[0,0], device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # When setting full_map for the first time, initialize prev_free_map
        if self.prev_free_map is None and self.full_map.shape[1] > 1:
            self.prev_free_map = torch.zeros_like(self.full_map[0, 1], device=self.device)

        

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_scenegraph(self):
        nodes = self.nodes
        edges = self.get_edges()
        caption_count = {}
        
        new_nodes = []
        
        node_id_map = {}
        for node in nodes:
            caption = node.caption
            if caption not in caption_count:
                caption_count[caption] = 0
            unique_id = "{}_{}".format(caption, caption_count[caption])
            new_nodes.append({
                'id': unique_id,
                'position': node.center
            })
            node_id_map[node] = unique_id
            caption_count[caption] += 1

        new_edges = []
        for edge in edges:
            source_id = node_id_map[edge.node1]
            target_id = node_id_map[edge.node2]
            new_edges.append({
                'source': source_id,
                'target': target_id,
                'type': edge.relation
            })

        self.scenegraph = {
            'nodes': new_nodes,
            'edges': new_edges
        }

        return self.scenegraph

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes

    def get_grounded_sam(self, device):
        model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, device=device)
        predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
        return model, predictor
    
    def get_segmentation(
        self, model, image: np.ndarray
    ) -> tuple:
        groundingdino = model[0]
        sam_predictor = model[1]
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
        boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
        if len(caption) == 0:
            return None, None, None, None
        sam_predictor.set_image(image)

        # size = image_pil.size
        H, W = image.shape[0], image.shape[1]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        mask, conf, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()
        return mask, xyxy, conf, caption

    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix

    def segment2d(self):
        print('    segement2d...')
        with torch.no_grad():
            print('        sam_segmentation...')
            mask, xyxy, masks_conf, caption = self.get_segmentation(self.grounded_sam, self.image_rgb)

            self.seg_xyxy = xyxy
            self.seg_caption = caption
            self.clear_line()
        if caption is None:
            self.clear_line()
            return
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=masks_conf,
            class_id=np.zeros_like(masks_conf).astype(int),
            mask=mask,
        )
        image_appear_efficiency = [''] * len(mask)
        self.segment2d_results.append({
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": self.classes,
            "image_appear_efficiency": image_appear_efficiency,
            "image_rgb": self.image_rgb,
            "caption": caption,
        })
        self.clear_line()


    def mapping3d(self):
        print('    mapping3d...')
        depth_array = self.image_depth
        depth_array = depth_array[..., 0]

        gobs = self.segment2d_results[-1]
        
        unt_pose = self.pose_matrix
        
        adjusted_pose = unt_pose
        cam_K = self.camera_matrix
            
        idx = len(self.segment2d_results) - 1

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = self.image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = adjusted_pose,
            class_names = self.classes,
            BG_CLASSES = self.BG_CLASSES,
            is_navigation = self.is_navigation
        )
        
        if len(fg_detection_list) == 0:
            self.clear_line()
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            self.clear_line()
            return
                
        print('        compute_spatial_similarities...')
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        self.clear_line()
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')
        
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim)
        
        self.objects_post = filter_objects(self.cfg, self.objects)
        self.clear_line()
            
    def get_caption(self):
        print('    get_caption...')
        for idx, object in enumerate(self.objects_post):
            caption_list = []
            for idx_det in range(len(object["image_idx"])):
                caption = self.segment2d_results[object["image_idx"][idx_det]]['caption'][object["mask_idx"][idx_det]]
                caption_list.append(caption)
            caption = self.find_modes(caption_list)[0]
            object['captions'] = [caption]
        self.clear_line()

    def update_node(self):
        print('    update_node...')
        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            # caption_new = self.find_modes(self.objects_post[i]['captions'])[0]
            caption_new = node.object['captions'][0]
            if caption_ori != caption_new:
                node.set_caption(caption_new)
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post))
        # for i in range(node_num_ori, node_num_new):
        for new_object in new_objects:
            new_node = ObjectNode()
            # caption = self.find_modes(self.objects_post[i]['captions'])[0]
            caption = new_object['captions'][0]
            new_node.set_caption(caption)
            new_node.set_object(new_object)
            # self.create_new_edge(new_node)
            self.nodes.append(new_node)
        # get node.center and node.room
        for node in self.nodes:
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)
            x = int(center[0] * 100 / self.map_resolution)
            y = int(center[1] * 100 / self.map_resolution)
            y = self.map_size - 1 - y
            node.set_center([x, y])
            if 0 <= x < self.map_size and 0 <= y < self.map_size and hasattr(self, 'room_map'):
                if sum(self.room_map[0, :, y, x]!=0).item() == 0:
                    room_label = 0
                else:
                    room_label = torch.where(self.room_map[0, :, y, x]!=0)[0][0].item()
            else:
                room_label = 0
            if node.room_node is not self.room_nodes[room_label]:
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)
        self.clear_line()

    def create_new_edge(self, new_node):
        # new_edges = []
        for j, old_node in enumerate(self.nodes):
            image = self.get_joint_image(old_node, new_node)
            if image is not None:
                response = self.vlm(self.prompt_create_relation.format(obj1=old_node.caption, obj2=new_node.caption), image)
                if "No clear spatial relationship" not in response:
                    response = response.lower()
                    objects = [old_node.caption.lower(), new_node.caption.lower()]
                    relations = self.graphbuilder.get_relations(response, objects)
                    new_edge = Edge(old_node, new_node)
                    if len(relations) > 0:
                        new_edge.set_relation(relations[0]['type'])
                    else:
                        new_edge.set_relation('')
                    # new_node.edges.add(new_edge)
                    # old_node.edges.add(new_edge)
                    # new_edges.append(new_edge)

    def update_edge_Deprecated(self):
        print('    update_edge...')
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        if len(new_nodes) == 0:
            self.clear_line()
            return
        # create the edge between new_node and old_node
        new_edges = []
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                # new_node.edges.add(new_edge)
                # old_node.edges.add(new_edge)
                new_edges.append(new_edge)
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                # new_node1.edges.add(new_edge)
                # new_node2.edges.add(new_edge)
                new_edges.append(new_edge)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals
        if len(new_edges) > 0:
            print(f'        LLM get all relation proposals...')
            node_pairs = []
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges)
            prompt = prompt.format(*node_pairs)
            relations = self.llm(prompt=prompt)
            relations = relations.split('\n')
            if len(relations) == len(new_edges):
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation)
            self.clear_line()
            # discriminate all relation proposals
            for i, new_edge in enumerate(new_edges):
                print(f'        discriminate_relation  {i}/{len(new_edges)}...')
                if new_edge.relation == None:
                    new_edge.delete()
                self.clear_line()
            # get edges set
            # self.edges = set()
            # for node in self.nodes:
            #     self.edges.update(node.edges)
        self.clear_line()

    def update_edge(self):
        print('    update_edge...')
        old_nodes = []
        new_nodes = [] # Nodes that are newly created or had their caption changed
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False # Reset the flag
            else:
                old_nodes.append(node)

        if len(new_nodes) == 0 and not any(node.is_new_node for node in old_nodes): # Added check for old_nodes that might have been re-flagged
            self.clear_line()
            return

        potential_new_edges = []
        # Edges between new nodes and existing old nodes
        for new_node in new_nodes:
            for old_node in old_nodes:
                edge = Edge(new_node, old_node)
                potential_new_edges.append(edge)
        
        # Edges between new nodes themselves
        for idx1, new_node1 in enumerate(new_nodes):
            for idx2 in range(idx1 + 1, len(new_nodes)): # Ensures idx1 < idx2
                new_node2 = new_nodes[idx2]
                edge = Edge(new_node1, new_node2)
                potential_new_edges.append(edge)
        
        # 如果 old_nodes 中也有 caption 更新（因此 is_new_node 可能为 True），也需要考虑它们之间的新边。
        # 但当前逻辑是 new_nodes 只包含新创建的或caption变化的，old_nodes 是未变化的。
        # 如果 caption 变化也让 old_node.is_new_node=True，那上面的 new_nodes 收集逻辑已经包含了它们。

        edges_for_llm_fallback = []
        vlm_processed_count = 0
        # 1. 优先尝试VLM和图像上下文
        print(f'        Attempting VLM for {len(potential_new_edges)} potential new edges...')
        for edge in potential_new_edges:
            node1, node2 = edge.node1, edge.node2
            image = self.get_joint_image(node1, node2) # 获取两个节点同时可见的图像

            if image is not None:
                try:
                    vlm_prompt = self.prompt_create_relation.format(obj1=node1.caption, obj2=node2.caption)
                    # self.vlm 是 VLM 类的实例
                    response = self.vlm(vlm_prompt, image)

                    if "No clear spatial relationship" not in response.lower():
                        # graphbuilder.get_relations 解析VLM的输出
                        parsed_relations = self.graphbuilder.get_relations(response.lower(), [node1.caption.lower(), node2.caption.lower()])
                        if parsed_relations:
                            edge.set_relation(parsed_relations[0]['type'])
                            vlm_processed_count += 1
                            print(f"          VLM found relation for ({node1.caption}, {node2.caption}): {edge.relation}")
                        else:
                            # VLM给出了回应，但无法解析为已知的关系类型
                            edges_for_llm_fallback.append(edge)
                    else:
                        # VLM明确表示没有清晰的空间关系
                        print(f"          VLM: No clear spatial relationship for ({node1.caption}, {node2.caption})")
                        # 你可以选择在此设置一个特殊的关系标签，或者也将其加入LLM回退列表
                        # edge.set_relation("no_vlm_relation") # 示例
                        edges_for_llm_fallback.append(edge)
                except Exception as e:
                    print(f"          Error during VLM processing for edge ({node1.caption}, {node2.caption}): {e}")
                    edges_for_llm_fallback.append(edge)
            else:
                # 没有找到共同图像，加入LLM回退列表
                edges_for_llm_fallback.append(edge)
        
        print(f'        VLM processed {vlm_processed_count} edges.')

        # 2. 对剩余的边使用LLM（可加入空间信息）
        if len(edges_for_llm_fallback) > 0:
            print(f'        LLM fallback for {len(edges_for_llm_fallback)} edges...')
            node_pairs_for_llm_prompt = []
            edges_in_llm_batch = []

            for edge in edges_for_llm_fallback:
                node1, node2 = edge.node1, edge.node2
                
                # 可选：计算3D空间距离作为提示信息
                # node.object['pcd'] 是点云数据
                # node.center 是2D地图坐标，使用3D信息更好
                try:
                    node1_pos_3d = np.asarray(node1.object['pcd'].points).mean(axis=0) # 世界坐标系
                    node2_pos_3d = np.asarray(node2.object['pcd'].points).mean(axis=0) # 世界坐标系
                    distance = np.linalg.norm(node2_pos_3d - node1_pos_3d)
                    spatial_hint = f" They are approximately {distance:.2f} meters apart."
                except Exception:
                    spatial_hint = "" # 如果计算失败则不加提示

                # 构建LLM的prompt部分
                # 注意：如果LLM不支持太长的batch prompt，或者每个prompt需要定制（比如加入复杂的空间关系描述），
                # 可能需要改为单独调用LLM。当前代码示例是批量调用。
                # self.prompt_edge_proposal 例子: "Object pair(s):"
                # 这里我们为每个pair构建一个条目
                node_pairs_for_llm_prompt.append(f"({node1.caption}, {node2.caption}){spatial_hint}")
                edges_in_llm_batch.append(edge)
            
            if node_pairs_for_llm_prompt:
                # 完整的prompt可能是 self.prompt_edge_proposal + "\n" + "\n".join(node_pairs_for_llm_prompt)
                # LLM期望的输入格式可能需要调整
                # 假设 self.llm 可以处理一个包含多个待判断对的列表，或者你需要循环调用
                # 以下是一个简化的批量prompt示例，实际LLM接口可能不同
                full_llm_prompt = self.prompt_edge_proposal + "\n" + "\n".join(node_pairs_for_llm_prompt)
                
                try:
                    llm_responses_str = self.llm(prompt=full_llm_prompt)
                    llm_relations_raw = llm_responses_str.split('\n')
                    # 过滤空行并去除首尾空格
                    llm_relations = [rel.strip() for rel in llm_relations_raw if rel.strip()]

                    if len(llm_relations) == len(edges_in_llm_batch):
                        for i, relation_text in enumerate(llm_relations):
                            edges_in_llm_batch[i].set_relation(relation_text.strip())
                            print(f"          LLM found relation for ({edges_in_llm_batch[i].node1.caption}, {edges_in_llm_batch[i].node2.caption}): {relation_text.strip()}")
                    else:
                        print(f"          LLM did not return the expected number of relations. Expected {len(edges_in_llm_batch)}, Got {len(llm_relations)}")
                        # 处理数量不匹配的情况，例如将其余的标记为未知或删除
                        for edge_to_delete in edges_in_llm_batch:
                            if edge_to_delete.relation is None: # 如果VLM和LLM都没有成功处理
                                print(f"            Deleting edge for ({edge_to_delete.node1.caption}, {edge_to_delete.node2.caption}) due to LLM mismatch/failure.")
                                edge_to_delete.delete()
                except Exception as e:
                    print(f"          Error during LLM processing: {e}")
                    for edge_to_delete in edges_in_llm_batch: # LLM调用失败，删除这些边
                         if edge_to_delete.relation is None:
                            print(f"            Deleting edge for ({edge_to_delete.node1.caption}, {edge_to_delete.node2.caption}) due to LLM error.")
                            edge_to_delete.delete()


        # 3. 清理没有成功建立关系的边 (可选，取决于你如何标记未成功处理的边)
        # 如果某些边经过VLM和LLM后 relation 仍然为 None，可以选择删除它们
        # 注意：Edge的构造函数会将其加入到对应节点的edges集合中，delete方法会将其移除
        all_edges_after_update = set()
        for node in self.nodes:
            all_edges_after_update.update(node.edges)
        
        for edge in list(all_edges_after_update): # 使用list副本进行迭代删除
            if edge.relation is None:
                print(f"        Final cleanup: Deleting edge ({edge.node1.caption}, {edge.node2.caption}) with no relation.")
                edge.delete()
        
        self.clear_line()

    def update_group(self):
        for room_node in self.room_nodes:
            if len(room_node.nodes) > 0:
                room_node.group_nodes = []
                object_nodes = list(room_node.nodes)
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=10, min_samples=1)  
                clusters = dbscan.fit_predict(centers)  
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.get_graph()
                    room_node.group_nodes.append(group_node)
    
    def insert_goal(self, goal=None):
        if goal is None:
            goal = self.obj_goal
        self.update_group()
        room_node_text = ''
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0:
                room_node_text = room_node_text + room_node.caption + ','
        # room_node_text[-2] = '.'
        if room_node_text == '':
            return None
        prompt = self.prompt_room_predict.format(goal, room_node_text)
        response = self.llm(prompt=prompt)
        response = response.lower()
        predict_room_node = None
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
                predict_room_node = room_node
        if predict_room_node is None:
            return None
        for group_node in predict_room_node.group_nodes:
            corr_score = self.graph_corr(goal, group_node)
            group_node.corr_score = corr_score
        sorted_group_nodes = sorted(predict_room_node.group_nodes)
        self.mid_term_goal = sorted_group_nodes[-1].center
        return self.mid_term_goal
    
    def update_scenegraph(self):
        print(f'update_observation {self.navigate_steps}...')
        self.segment2d()
        if len(self.segment2d_results) == 0:
            self.clear_line()
            return
        self.mapping3d()
        self.get_caption()
        self.update_node()
        self.update_edge_Deprecated()
        # self.update_edge()
        self.get_scenegraph()
        
        # >>> New: Update semantic map and visualize at each step <<<
        # This is the core entry point for integrating new functionality.
        print("    Updating semantic score map...")
        self._update_semantic_score_map()
        print("    Visualizing semantic map...")
        self.visualize_semantic_map()
        self.clear_line(3) # Clear three lines of output
        # >>> End of new additions <<<

    def explore(self):
        # >>> Modified: Refactor explore method to integrate adaptive strategy <<<
        # Core idea: When high-level semantic graph matching cannot provide clear guidance,
        # instead of degrading to simple exploration, launch adaptive exploration module based on VLM and TSP.
        
        overlap = self.overlap()
        
        # Keep the original high matching logic unchanged
        if 0.5 <= overlap < 0.9 and len(self.matcher.common_nodes) >= 2:
            print("UniGoal Stage 3 (High Matching): Exploring remaining subgraph nodes.")
            goal = self.explore_remaining()
        elif overlap >= 0.9 and len(self.matcher.common_nodes) < 2:
            print("UniGoal Stage 4 (Very High Matching): Correcting scene graph reasonableness.")
            goal = self.reasonableness_correction()
        else:
            # When matching is low, call the packaged adaptive exploration strategy, which is the core of this modification.
            # This replaces the original explore_subgraph() or simpler greedy search.
            print(f"UniGoal Stage 1/2 (Low Matching, Overlap: {overlap:.2f}): Switching to Adaptive Exploration.")
            goal = self._adaptive_explore()
        
        # get_goal will select a final reachable target based on the calculated goal point (possibly None) and current frontiers.
        goal = self.get_goal(goal)
        
        return goal
        # >>> End of modifications <<<

    def explore_subgraph(self, goal=None):
        if goal == None:
            self.subgraph = self.goalgraph_decomposed['subgraph_1']
        self.subgraph = self.goalgraphdecomposer.graph_to_text(self.subgraph)
        return self.insert_goal(self.subgraph)
    
    def overlap(self):
        graph1 = self.scenegraph
        graph2 = self.goalgraph
        self.matcher = GraphMatcher(graph1, graph2, self.llm)
        overlap_score = self.matcher.overlap()
        return overlap_score
    
    def explore_remaining(self):
        G1 = self.matcher.G1
        G2 = self.matcher.G2
        common_nodes = self.matcher.common_nodes

        # Assign positions to the first two common nodes in the subgraph
        for i, node_id in enumerate(common_nodes):
            if i < 2:
                G2.nodes[node_id]['position'] = G1.nodes[node_id]['position']
            else:
                break

        # Calculate relative positions within the subgraph
        positions = self.matcher.calculate_relative_positions(G2, common_nodes)

        # Predict positions of the remaining nodes
        position = self.matcher.predictconda_remaining_node_positions(common_nodes, positions, G1)
        return position

    def reasonableness_correction(self):
        corrector = SceneGraphCorrector(self.llm)
        self.scenegraph = corrector.correct_scene_graph(self.scenegraph, self.obj_goal)
        return None

    def clear_line(self, line_num=1):
        for i in range(line_num):  
            sys.stdout.write('\033[F')
            sys.stdout.write('\033[J')
            sys.stdout.flush()  
    
    def find_modes(self, lst):  
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)  
            max_count = max(counts.values())  
            modes = [item for item, count in counts.items() if count == max_count]  
            return modes  
        
    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        return image

    def get_goal(self, goal=None):
        fbe_map = torch.zeros_like(self.full_map[0,0])
        if self.full_map.shape[1] == 1:
            fbe_map[self.fbe_free_map[0,0]>0] = 1
        else:
            fbe_map[self.full_map[0,1]>0] = 1
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_map = remove_small_frontiers(frontier_map, min_size=20)
        # >>> Modified: Update self.frontier_locations attribute <<<
        # Convert tensor to numpy array and store frontier point positions
        frontier_locations_tensor = torch.stack(torch.where(frontier_map)).T
        self.frontier_locations = frontier_locations_tensor.cpu().numpy() # Store as Nx2 numpy array

        num_frontiers = len(self.frontier_locations)
        if num_frontiers == 0:
            return None
        
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose[0].cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, start = self.get_traversible(self.full_map.cpu().numpy()[0, 0, ::-1], input_pose)
        
        # >>> Modified: Update information required by FMM planner <<<
        self.traversible = traversible
        self.start = start
        
        # If `goal` (provided by adaptive strategy) is not None, use it directly.
        # If None (e.g., all exploration strategies failed), fall back to original distance-based greedy selection.
        if goal is not None and (isinstance(goal, list) or isinstance(goal, np.ndarray)):
            # This case is when adaptive exploration strategy provides a clear target point.
            # We directly return this point. Note that the original function has complex weighted scoring afterwards,
            # but to simplify and follow the new strategy, we directly adopt the decision from `_adaptive_explore`.
            # `goal` is already in map coordinates, no need to convert.
            return goal

        # --- The following is the fallback logic when `goal` is None (original greedy logic) ---
        planner = FMMPlanner(traversible)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        
        # Note: frontier_locations here is numpy array
        frontier_locations_fmm = self.frontier_locations + 1
        # Extract distances
        distances = fmm_dist[frontier_locations_fmm[:, 0], frontier_locations_fmm[:, 1]] / 20

        distance_threshold = 1.2
        idx_valid = np.where(distances >= distance_threshold)[0]
        if len(idx_valid) == 0:
            # If no frontiers meet the distance threshold, choose the closest one
            if len(distances) > 0:
                closest_idx = np.argmin(distances)
                return self.frontier_locations[closest_idx]
            return None

        distances_valid = distances[idx_valid]
        distances_valid_inverse = 10 - (np.clip(distances_valid, 0, 10 + distance_threshold) - distance_threshold)
        
        # Choose the one with the highest score
        idx_max_in_valid = np.argmax(distances_valid_inverse)
        final_idx_in_original = idx_valid[idx_max_in_valid]
        goal = self.frontier_locations[final_idx_in_original]

        if self.start is not None:
            # If path history is empty, or current position is different from previous position, add it
            if not self.path_history or np.any(self.path_history[-1] != self.start):
                self.path_history.append(self.start)

        return goal

    def get_traversible(self, map_pred, pose_pred):
        if isinstance(map_pred, torch.Tensor):
            map_pred = map_pred.cpu().numpy()
        if len(map_pred.shape) == 4:
            map_pred = map_pred[0, 0]
        grid = np.rint(map_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        # start = [int(start_x), int(start_y)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        #Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        selem = skimage.morphology.square(1)
        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        # obstacle dilation do not dilate collision
        traversible = 1 - traversible
        selem = skimage.morphology.disk(4)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start
    
    def reset(self):
        full_width, full_height = self.map_size, self.map_size
        self.full_width = full_width
        self.full_height = full_height
        self.visited = torch.zeros(full_width, full_height).float().cpu().numpy()
        self.segment2d_results = []
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.group_nodes = []
        self.init_room_nodes()
        self.edge_list = []
        # >>> New: Clear semantic maps when resetting <<<
        self.semantic_score_map = None
        self.semantic_confidence_map = None
        self.prev_free_map = None
        # Free space map for calculating newly visible areas
        self.prev_free_map = None
        # Path recording
        self.tsp_path_info = None
                # >>> New: Clear path history when resetting <<<
        if hasattr(self, 'path_history'):
            self.path_history.clear()

                # >>> New: Reset VLM context <<<
        self.vlm_context_summary = "Mission has just started. I need to explore the environment to find the target."
        self.vlm_short_term_memory.clear()
        self.vlm_found_goal = False
        self.vlm_goal_confidence = 0.0
        self.vlm_update_counter = 0


    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.llm(prompt=prompt)
        prompt = self.prompt_graph_corr_3.format(response_0, response_1 + response_2, graph.center_node.caption, goal)
        response_3 = self.llm(prompt=prompt)
        corr_score = self.text2value(response_3)
        return corr_score
    
    def text2value(self, text):
        try:
            value = float(text)
        except:
            value = 0
        return value
    
    # ===================================================================
    # >>> New: Core refactoring section - All new functions required for adaptive exploration <<<
    # ===================================================================

    def _adaptive_explore(self):
        """
        >>> New <<<
        Adaptive exploration strategy that encapsulates ApexNav ideas.
        This strategy dynamically selects exploration modes based on the strength of environmental semantic cues.

        Returns:
            np.ndarray: Calculated next target point coordinates (y, x), returns None if unable to decide.
        """
        # 1. Ensure we have a target subgraph as the basis for VLM queries
        if not hasattr(self, 'subgraph') or self.subgraph is None:
             if self.goalgraph_decomposed and 'subgraph_1' in self.goalgraph_decomposed:
                 self.subgraph = self.goalgraph_decomposed['subgraph_1']
             else:
                 # If no decomposed subgraph, use the entire goal graph as exploration basis
                 self.subgraph = self.goalgraph

        # 2. Calculate semantic scores for all frontier points
        frontier_scores = self._calculate_frontier_semantic_scores()
        
        # 3. If no valid frontiers or scores, fall back to simplest geometric exploration
        if frontier_scores is None or len(frontier_scores) == 0:
            print("Adaptive Strategy: No valid frontiers or scores. Falling back to geometric exploration.")
            return self._explore_geometric()

        # 4. Calculate score distribution metrics to judge semantic cue strength
        s_max = np.max(frontier_scores)
        s_mean = np.mean(frontier_scores)
        s_std = np.std(frontier_scores)
        
        # Ensure s_mean is not zero to avoid division by zero
        ratio = s_max / s_mean if s_mean > 1e-6 else 0

        # 5. Read from configuration or use default thresholds
        r_threshold = getattr(self.args, 'adaptive_r_threshold', 1.10)
        sigma_threshold = getattr(self.args, 'adaptive_sigma_threshold', 0.015)

        # 6. Make decisions based on metrics and call corresponding exploration modes
        if ratio > r_threshold and s_std > sigma_threshold:
            # Strong semantic cues -> Execute TSP-based semantic exploration
            print(f"Adaptive Strategy: Strong semantic cues (Ratio: {ratio:.2f} > {r_threshold}, StdDev: {s_std:.3f} > {sigma_threshold}). Using TSP-based Semantic Exploration.")
            goal = self._explore_semantic_tsp(frontier_scores)
        else:
            # Weak semantic cues -> Execute geometry-based exploration
            print(f"Adaptive Strategy: Weak semantic cues (Ratio: {ratio:.2f}, StdDev: {s_std:.3f}). Using Geometric Exploration.")
            goal = self._explore_geometric()
            
        return goal

    def _calculate_frontier_semantic_scores(self):
        """
        >>> New <<<
        Extract semantic scores for each frontier point from the generated `semantic_score_map`.
        
        Returns:
            list[float] or None: List of semantic scores for each frontier point, or None if no frontiers.
        """
        if self.frontier_locations is None or len(self.frontier_locations) == 0:
            print("Semantic Scoring: No frontiers found.")
            return None

        frontier_scores = []
        # Convert PyTorch tensor to NumPy array for easier indexing
        semantic_map_np = self.semantic_score_map.cpu().numpy()[0, 0]

        # Iterate through all frontier points and query their scores on the semantic map
        for loc in self.frontier_locations:
            y, x = loc # Note: coordinate order is (y, x)
            # Ensure coordinates are within map bounds
            if 0 <= y < semantic_map_np.shape[0] and 0 <= x < semantic_map_np.shape[1]:
                score = semantic_map_np[y, x]
                frontier_scores.append(score)
            else:
                # If frontier point is outside map (theoretically shouldn't happen), give 0 score
                frontier_scores.append(0.0)
                
        return frontier_scores

    def _explore_geometric(self):
        """
        >>> Modified version <<<
        Geometry-based exploration: Select a frontier point that guides the agent away from recent trajectories.
        Suitable for environments with weak semantic cues, aiming to efficiently expand known areas and avoid repetitive exploration.

        Returns:
            np.ndarray or None: Calculated optimal frontier point coordinates (y, x), or None if unable to plan.
        """
        if self.traversible is None or self.start is None:
            print("Geometric Exploration: Traversible map or start pose not available.")
            return None

        if self.frontier_locations is None or len(self.frontier_locations) == 0:
            print("Geometric Exploration: No frontiers to explore.")
            return None

        # Use FMMPlanner to calculate traversible distances from agent to all points
        planner = FMMPlanner(self.traversible, step_size=5)
        start_pose_fmm = (self.start[0] + 1, self.start[1] + 1)
        planner.set_goal(start_pose_fmm)
        fmm_dist = planner.fmm_dist

        # Get distances to all frontier points
        frontier_locations_fmm = self.frontier_locations + 1
        distances = fmm_dist[frontier_locations_fmm[:, 0], frontier_locations_fmm[:, 1]]
        
        # Filter out unreachable frontier points
        reachable_mask = distances < 1e5
        if not np.any(reachable_mask):
            return None # No reachable frontier points

        reachable_frontiers = self.frontier_locations[reachable_mask]
        reachable_distances = distances[reachable_mask]

        # --- New logic: Penalize frontier points near recent path ---
        # If path history is not long enough, use original "nearest point" strategy to avoid early behavior anomalies
        if len(self.path_history) < 10:
            print("Geometric Exploration: Path history is short, choosing nearest frontier.")
            min_dist_idx = np.argmin(reachable_distances)
            best_frontier = reachable_frontiers[min_dist_idx]
            return best_frontier

        print("Geometric Exploration: Penalizing frontiers near recent path to avoid repetition.")
        history_points = np.array(list(self.path_history))
        
        # Efficiently calculate Euclidean distance matrix from each reachable frontier to all points in "historical path"
        dist_to_history = cdist(reachable_frontiers, history_points)
        # Find the "nearest" distance from each frontier point to historical path
        min_dist_to_history = np.min(dist_to_history, axis=1)
        
        # Define penalty parameters
        penalty_radius = 25.0  # Penalty radius (pixels), frontier points within this distance from path are penalized
        penalty_strength = 10000.0 # A huge penalty value to ensure penalized points are not selected
        
        # Calculate penalty scores
        path_penalty = np.zeros_like(reachable_distances)
        path_penalty[min_dist_to_history < penalty_radius] = penalty_strength
        
        # Add original "walking distance" and "path penalty" to get final score
        combined_scores = reachable_distances + path_penalty
        
        # [Robustness handling] If all reachable frontier points are penalized (e.g., in narrow corridors)
        # Ignore penalty and choose the nearest point to avoid getting stuck
        if np.all(path_penalty > 0):
            print("Geometric Exploration: All frontiers are near path. Ignoring penalty to avoid getting stuck.")
            min_dist_idx = np.argmin(reachable_distances)
            best_frontier = reachable_frontiers[min_dist_idx]
        else:
            # Choose the frontier point with lowest combined score (close distance and far from old path)
            min_score_idx = np.argmin(combined_scores)
            best_frontier = reachable_frontiers[min_score_idx]
            
        return best_frontier

    def _explore_semantic_tsp(self, frontier_scores):
        """
        >>> Modified version <<<
        TSP-based semantic exploration: Plan optimal access paths between high-value frontier points.
        Added clustering preprocessing step to select more diverse and efficient exploration targets.
        """
        if self.traversible is None or self.start is None:
            return self._explore_geometric()

        scores = np.array(frontier_scores)
        
        # 1. Filter out all frontier points with scores above average
        mean_score = np.mean(scores)
        high_score_indices = np.where(scores > mean_score)[0]

        if len(high_score_indices) == 0:
            return self._explore_geometric() # If no high-score points, fall back to geometric exploration

        high_score_frontiers = self.frontier_locations[high_score_indices]
        high_scores = scores[high_score_indices]

        # 2. >>> New: Cluster high-score frontier points <<<
        print(f"  TSP: Found {len(high_score_frontiers)} high-score frontiers. Clustering them...")
        # 1.5 meter radius, when map resolution is 5cm/px, radius is 30 pixels
        clustering_radius_px = 1.5 * 100 / self.map_resolution 
        clustered_frontiers, clustered_scores = self._cluster_frontiers(
            high_score_frontiers, high_scores, radius_px=clustering_radius_px
        )
        print(f"  TSP: Clustered down to {len(clustered_frontiers)} representative frontiers.")

        # 3. From clustered representative points, select the top-K highest scoring ones as final TSP candidates
        MAX_TSP_CITIES = 10 # Can appropriately reduce TSP city count after clustering
        if len(clustered_frontiers) > MAX_TSP_CITIES:
            # np.argsort returns indices from small to large, we take the last K
            top_k_indices = np.argsort(clustered_scores)[-MAX_TSP_CITIES:]
            tsp_frontiers = clustered_frontiers[top_k_indices]
        else:
            tsp_frontiers = clustered_frontiers

        if len(tsp_frontiers) < 2:
            self.tsp_path_info = None # Clear old path
            if len(tsp_frontiers) == 1:
                return tsp_frontiers[0] # If only one representative point, set as target directly
            else:
                return self._explore_geometric() # If no points after clustering, fall back

        # 4. Subsequent TSP calculation logic (cost matrix, solving, etc.) remains unchanged, just operates on clustered points
        N = len(tsp_frontiers)
        cost_matrix = np.full((N + 1, N + 1), 1e6)
        
        points = [(self.start[0] + 1, self.start[1] + 1)] + \
                 [(f[0] + 1, f[1] + 1) for f in tsp_frontiers]
        
        planner = FMMPlanner(self.traversible, step_size=5)

        print(f"  TSP: Building cost matrix for {N+1} points...")
        for i in range(N + 1):
            # ... (All subsequent code is identical to your original version)
            # ... (Calculate cost_matrix, solve_tsp_dynamic_programming, return goal)
            start_point = points[i]
            planner.set_goal(start_point)
            fmm_dist = planner.fmm_dist

            for j in range(N + 1):
                if i == j:
                    cost_matrix[i, j] = 0
                    continue
                end_point = points[j]
                if 0 <= end_point[0] < fmm_dist.shape[0] and 0 <= end_point[1] < fmm_dist.shape[1]:
                    dist = fmm_dist[end_point[0], end_point[1]]
                    if dist < 1e5:
                        cost_matrix[i, j] = dist
        
        print("  TSP: Solving...")
        try:
            permutation, _ = solve_tsp_dynamic_programming(cost_matrix)
        except Exception as e:
            print(f"TSP solver failed: {e}. Falling back to geometric exploration.")
            self.tsp_path_info = None
            return self._explore_geometric()

        if not permutation or len(permutation) < 2:
            self.tsp_path_info = None
            return self._explore_geometric()
        
        # --- (Visualization and goal return parts also remain unchanged) ---
        tsp_points_indices_in_high_score_list = [p - 1 for p in permutation if p > 0]
        self.tsp_path_info = {
            "frontiers": tsp_frontiers,
            "path_indices": tsp_points_indices_in_high_score_list,
            "full_path_coords": np.vstack([self.start, tsp_frontiers[tsp_points_indices_in_high_score_list]])
        }

        next_point_index_in_tsp_list = permutation[1]
        next_frontier_idx_in_high_score_list = next_point_index_in_tsp_list - 1
        
        goal = tsp_frontiers[next_frontier_idx_in_high_score_list]
        
        return goal

    def _update_semantic_score_map(self):
        """
        >>> Modified (Core optimization V3) <<<
        Implement dual-path VLM interaction:
        1. If there is a target image, perform visual comparison scoring.
        2. If only text target, fall back to optimized text prompt scoring.
        """
        if not hasattr(self, 'image_rgb') or self.image_rgb is None or \
           self.full_map is None or self.full_map.shape[1] < 2 or \
           self.prev_free_map is None:
            return

        rgb_image_pil = Image.fromarray(self.image_rgb)

        # 1. Pose a standard question
        question = f"Based on my current view, how relevant is it for finding '{self.text_goal}'? And do you see the target?"

        # 2. Call new VLM query function with context
        vlm_result = self._query_vlm_with_context(rgb_image_pil, question)
        
        # 3. Extract semantic score from structured result
        current_score = vlm_result.get('semantic_score', 0.5)
        print(f"  VLM Judgement: Reason='{vlm_result.get('reason', 'N/A')}', Score={current_score:.2f}, GoalFound={self.vlm_found_goal}, Confidence={self.vlm_goal_confidence:.2f}")


        current_score = max(0.0, min(1.0, current_score))
        current_free_map = self.full_map[0, 1]
        newly_visible_mask = (current_free_map > 0) & (self.prev_free_map == 0)

        if not newly_visible_mask.any():
            self.prev_free_map = current_free_map.clone()
            return

        y_coords, x_coords = torch.where(newly_visible_mask)
        current_confidence = 0.8
        prev_scores = self.semantic_score_map[0, 0, y_coords, x_coords]
        prev_confidences = self.semantic_confidence_map[0, 0, y_coords, x_coords]
        new_confidences = prev_confidences + current_confidence
        safe_new_confidences = torch.where(new_confidences > 1e-6, new_confidences, torch.tensor(1.0, device=self.device))
        new_scores = (prev_scores * prev_confidences + current_score * current_confidence) / safe_new_confidences
        self.semantic_score_map[0, 0, y_coords, x_coords] = new_scores
        self.semantic_confidence_map[0, 0, y_coords, x_coords] = new_confidences
        self.prev_free_map = current_free_map.clone()


    def visualize_semantic_map(self):
        """
        >>> Modified <<<
        Render the current semantic score map as a colored heatmap and overlay it with the obstacle map.
        Returns a NumPy image array instead of saving to file.
        """
        if self.semantic_score_map is None:
            return np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        score_map_np = self.semantic_score_map[0, 0].cpu().numpy()
        
        heatmap_gray = (score_map_np * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        
        if self.full_map is None or self.full_map.shape[1] < 2:
             return heatmap_color # If no map, only return heatmap
            
        obstacle_map_np = self.full_map[0, 0].cpu().numpy()
        free_map_np = self.full_map[0, 1].cpu().numpy()
        
        background = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        background[free_map_np > 0] = [255, 255, 255]
        background[obstacle_map_np == 1] = [100, 100, 100]

        mask = (score_map_np > 0.01) & (free_map_np > 0)
        
        if np.any(mask):
            alpha = 0.6
            beta = 1.0 - alpha
            background[mask] = cv2.addWeighted(heatmap_color[mask], alpha, background[mask], beta, 0)

        if self.start is not None:
            try:
                agent_y, agent_x = int(self.start[0]), int(self.start[1])
                cv2.circle(background, (agent_x, agent_y), radius=5, color=(0, 0, 255), thickness=-1)
            except (TypeError, ValueError):
                pass
        
        # Return flipped image to match coordinate system
        return cv2.flip(background, 0)
    
    def _create_stitched_image(self, goal_image: Image.Image, current_view_image: Image.Image, gap=10) -> Image.Image:
        """
        Horizontally stitch the goal image and current view image into one image.
        """
        # Unify the height of both images
        height = max(goal_image.height, current_view_image.height)
        goal_image = goal_image.resize((int(goal_image.width * height / goal_image.height), height))
        current_view_image = current_view_image.resize((int(current_view_image.width * height / current_view_image.height), height))

        # Create new canvas
        stitched_width = goal_image.width + current_view_image.width + gap
        stitched_image = Image.new('RGB', (stitched_width, height), (255, 255, 255))

        # Paste images
        stitched_image.paste(goal_image, (0, 0))
        stitched_image.paste(current_view_image, (goal_image.width + gap, 0))

        return stitched_image

    def _cluster_frontiers(self, frontiers, scores, radius_px, min_samples=1):
        """
        >>> New helper function <<<
        Use DBSCAN to cluster frontier points to reduce redundancy.

        Parameters:
            frontiers (np.ndarray): Nx2 array of frontier point coordinates.
            scores (np.ndarray): N array of frontier point scores.
            radius_px (float): DBSCAN clustering radius (eps) in pixels.
            min_samples (int): Minimum number of points required to form a cluster.

        Returns:
            (np.ndarray, np.ndarray): Coordinates and scores of representative points after clustering.
        """
        if len(frontiers) == 0:
            return np.array([]), np.array([])

        # Execute DBSCAN clustering
        db = DBSCAN(eps=radius_px, min_samples=min_samples).fit(frontiers)
        labels = db.labels_

        # Find representative points for each cluster (select the highest scoring point in that cluster)
        representative_frontiers = []
        representative_scores = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                # -1 represents noise points, we treat each noise point as an independent cluster
                noise_indices = np.where(labels == -1)[0]
                for idx in noise_indices:
                    representative_frontiers.append(frontiers[idx])
                    representative_scores.append(scores[idx])
                continue

            # Find all points in current cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_scores = scores[cluster_indices]
            
            # In current cluster, find the index of the point with highest score
            highest_score_in_cluster_idx = cluster_indices[np.argmax(cluster_scores)]
            
            # Use this highest scoring point as representative for the entire cluster
            representative_frontiers.append(frontiers[highest_score_in_cluster_idx])
            representative_scores.append(scores[highest_score_in_cluster_idx])

        return np.array(representative_frontiers), np.array(representative_scores)


    def _update_vlm_context(self):
        """
        >>> New <<<
        Use LLM to distill short-term memory into long-term summary.
        """
        # Trigger summary update whenever short-term memory is full
        if len(self.vlm_short_term_memory) == self.vlm_short_term_memory.maxlen:
            print("  VLM Context: Summarizing short-term memory...")
            
            # Convert short-term memory to text
            history_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in self.vlm_short_term_memory])
            
            # Create a prompt for summarization
            summarization_prompt = f"""
You are a navigation robot's memory assistant.
Below is the robot's recent interaction history and its previous mission summary.
Your task is to create a new, concise mission summary (no more than two sentences).
The summary should include what has been explored, what key objects have been found, and what the current main objective is.

Previous Summary: "{self.vlm_context_summary}"

Recent History:
{history_text}

New concise mission summary:
"""
            # Call LLM (not VLM) for summarization
            try:
                new_summary = self.llm(prompt=summarization_prompt)
                self.vlm_context_summary = new_summary.strip()
                print(f"  VLM Context: New summary is '{self.vlm_context_summary}'")
                # Clear short-term memory to avoid redundant summarization
                self.vlm_short_term_memory.clear()
            except Exception as e:
                print(f"  VLM Context: Failed to summarize memory due to an error: {e}")

    def _query_vlm_with_context(self, image_pil, question):
        """
        >>> New <<<
        A structured VLM query function with context and state injection.
        """
        # 1. Prepare agent state information
        current_room = "an unknown room"
        # Find the room where the agent is currently located
        if self.start:
            y, x = int(self.start[0]), int(self.start[1])
            if hasattr(self, 'room_map') and 0 <= y < self.map_size and 0 <= x < self.map_size:
                room_idx_tensor = torch.where(self.room_map[0, :, y, x] != 0)[0]
                if len(room_idx_tensor) > 0:
                    room_idx = room_idx_tensor[0].item()
                    if room_idx > 0 and room_idx < len(self.rooms):
                        current_room = self.rooms[room_idx-1] # Assume room_map index starts from 1
        
        # Simplify scene graph information
        nearby_objects = [node.caption for node in self.nodes[-5:]] # Last 5 objects seen
        state_info = f"""
[Agent Status]
Current Location: I am in {current_room}.
Current Target: I am looking for "{self.text_goal}".
Nearby Objects Seen: {', '.join(nearby_objects) if nearby_objects else 'None yet'}.
"""
        
        # 2. Prepare historical context information
        short_term_history = "\n".join([f"Previous Q: {item['question']}\nPrevious A: {item['answer']}" for item in self.vlm_short_term_memory])

        # 3. Build final prompt
        # This is the core of this modification, we require VLM to perform multi-task output
        final_prompt = f"""
You are an intelligent assistant for a robot navigating a house.
Analyze the current image based on the mission context and agent status provided.
Respond in a strict JSON format with four keys: "reason", "semantic_score", "goal_found", "confidence".

1.  `reason`: (string) A brief explanation for your analysis.
2.  `semantic_score`: (float, 0.0 to 1.0) How relevant is the current view to finding the target? 1.0 means the target is likely very close.
3.  `goal_found`: (boolean) Is the specific target "{self.text_goal}" visible and identifiable in the current image?
4.  `confidence`: (float, 0.0 to 1.0) Your confidence level in the "goal_found" judgment.

---
[Mission Context]
{self.vlm_context_summary}
{state_info}
{short_term_history}
---

[Current Task]
Question: {question}
Now, analyze the provided image and provide your response in the specified JSON format.
"""

        # 4. Call VLM and parse results
        response_text = ""
        try:
            response_text = self.vlm(final_prompt, image_pil).strip()
            # Try to parse JSON
            parsed_json = json.loads(response_text.replace("```json", "").replace("```", "").strip())
            
            # 5. Update agent state
            self.vlm_found_goal = parsed_json.get('goal_found', False)
            self.vlm_goal_confidence = parsed_json.get('confidence', 0.0)
            
            # 6. Update context history
            self.vlm_short_term_memory.append({
                "question": question,
                "answer": parsed_json.get('reason', 'No reason provided.')
            })
            
            # 7. Periodically trigger long-term summary update
            self._update_vlm_context()

            return parsed_json

        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            print(f"  VLM Error: Could not parse VLM's JSON response. Error: {e}")
            print(f"  VLM Raw Response: {response_text}")
            # Return a default value to prevent system crash
            return {
                "reason": "VLM response was not valid JSON.",
                "semantic_score": 0.5, # Neutral score
                "goal_found": False,
                "confidence": 0.0
            }
    # ===================================================================
    # >>> End of new functions <<<
    # ===================================================================