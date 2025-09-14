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
# 引入 python-tsp 库用于解决旅行商问题
from python_tsp.exact import solve_tsp_dynamic_programming


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
"{obj1} and {obj2}: {obj1} is <relation type> {obj2}"
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

        # --- 新增：为自适应探索策略和TSP添加属性 ---
        # 存储计算出的边界点坐标 (x, y)
        self.frontier_locations = None
        # 当前子图目标，用于在不同探索策略间共享
        self.subgraph = None
        # FMM Planner需要一个可行走区域的地图
        self.traversible = None
        # 智能体在可行走地图中的起始坐标
        self.start = None

        # ------------------------------------------

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
        # 初始化一个与 full_map 同样大小的地图，用于存储VLM的语义得分
        self.semantic_score_map = torch.zeros_like(self.full_map)
        # 存储每个格子的置信度，用于加权平均
        self.semantic_confidence_map = torch.zeros_like(self.full_map)
        

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
        # --- 新增：每次更新时，都调用VLM来更新语义地图 ---
        self._update_semantic_score_map()
        # --- 新增：可视化语义地图 ---
        self.visualize_semantic_map()
        self.clear_line()

    def explore(self):
        overlap = self.overlap()
        if 0.5 <= overlap < 0.9 and len(self.matcher.common_nodes) >= 2:
            goal = self.explore_remaining()
        elif overlap >= 0.9 and len(self.matcher.common_nodes) < 2:
            goal = self.reasonableness_correction()
        else:
            # 当匹配度低时，调用封装好的自适应探索策略
            goal = self._adaptive_explore()
            # goal = self.explore_subgraph()
        
        goal = self.get_goal(goal)
        
        return goal

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
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return None
        
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, start = self.get_traversible(self.full_map.cpu().numpy()[0, 0, ::-1], input_pose)
        # --- 新增：存储FMM规划器所需的信息 ---
        self.traversible = traversible
        self.start = start
        # ------------------------------------
        planner = FMMPlanner(traversible)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        distance_threshold = 1.2
        idx_16 = np.where(distances>=distance_threshold)
        distances_16 = distances[idx_16]
        distances_16_inverse = 10 - (np.clip(distances_16, 0, 10 + distance_threshold) - distance_threshold)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175
        scores = np.zeros((num_16_frontiers))
        
        scores += distances_16_inverse
        if isinstance(goal, list) or isinstance(goal, np.ndarray):
            goal = list(goal)

            planner = FMMPlanner(traversible)
            state = [goal[0] + 1, goal[1] + 1]
            planner.set_goal(state)
            fmm_dist = planner.fmm_dist[::-1]
            distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
            
            distances_16 = distances[idx_16]
            distances_16_inverse = 1 - (np.clip(distances_16, 0, 10 + distance_threshold) - distance_threshold) / 10
            if len(distances_16) == 0:
                return None
            scores += distances_16_inverse

        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores
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
    # >>> 核心改造：新增自适应探索所需的核心辅助函数 <<<
    # ===================================================================

    def _calculate_frontier_semantic_scores(self):
        """
        从预先计算好的 self.semantic_score_map 中为每个边界点提取语义得分。
        这个函数现在变得非常高效，因为它只是一个查询操作。
        
        返回:
            list[float]: 每个边界点的语义得分列表。
        """
        # 检查是否存在边界点
        if not hasattr(self, 'frontier_locations') or self.frontier_locations is None or len(self.frontier_locations) == 0:
            print("Adaptive Strategy: No frontiers found.")
            return []

        frontier_scores = []
        semantic_map_np = self.semantic_score_map.cpu().numpy()[0, 0]

        # 遍历所有边界点，查询它们在语义地图上的得分
        for loc in self.frontier_locations:
            x, y = loc
            # 确保坐标在地图范围内
            if 0 <= x < semantic_map_np.shape[0] and 0 <= y < semantic_map_np.shape[1]:
                score = semantic_map_np[x, y]
                frontier_scores.append(score)
            else:
                # 如果边界点在地图外（理论上不应发生），给一个0分
                frontier_scores.append(0.0)
                
        return frontier_scores

    def _explore_geometric(self):
        """
        基于几何的探索：选择距离智能体当前位置最近的前沿点。
        适用于语义线索较弱的环境，目标是最高效地扩大探索区域。

        返回:
            np.ndarray: 最近边界点的坐标 (x, y)，如果没有则返回 None。
        """
        # 检查是否有可用于规划的地图和边界点
        if not hasattr(self, 'traversible') or self.traversible is None:
            print("Geometric Exploration: Traversible map not available.")
            return None

        if not hasattr(self, 'frontier_locations') or self.frontier_locations is None or len(self.frontier_locations) == 0:
            print("Geometric Exploration: No frontiers to explore.")
            return None

        # 使用 FMMPlanner 计算从智能体当前位置到所有点的距离场
        planner = FMMPlanner(self.traversible)
        
        # FMMPlanner内部会处理边界，所以坐标需要+1
        start_pose_fmm = (self.start[0] + 1, self.start[1] + 1)
        
        # 设置目标点为智能体当前位置，从而计算出距离场
        planner.set_goal(start_pose_fmm)
        fmm_dist = planner.fmm_dist[::-1] # FMM的y轴与地图的y轴相反，需要翻转

        # 准备边界点坐标 (同样需要+1)
        frontier_locations_fmm = self.frontier_locations + 1
        
        # 获取每个边界点到智能体的距离
        # 使用clip确保索引在范围内
        valid_frontiers = []
        valid_distances = []
        for loc in frontier_locations_fmm:
            x, y = loc
            if 0 <= x < fmm_dist.shape[0] and 0 <= y < fmm_dist.shape[1]:
                valid_frontiers.append(loc)
                valid_distances.append(fmm_dist[x, y])
        
        if not valid_distances:
            return None
            
        # 找到最小距离的索引
        min_dist_idx = np.argmin(valid_distances)
        
        # 获取最近边界点的坐标 (记得-1恢复到原始地图坐标)
        nearest_frontier = valid_frontiers[min_dist_idx] - 1
        
        return nearest_frontier

    def _explore_semantic_tsp(self, frontier_scores):
        """
        基于TSP的语义探索：在高价值边界点之间规划最优路径。

        Args:
            frontier_scores (list[float]): 每个边界点的语义得分。

        返回:
            np.ndarray: TSP路径上的第一个边界点坐标。
        """
        if not hasattr(self, 'traversible') or self.traversible is None:
            return self._explore_geometric() # 没有地图则退化

        scores = np.array(frontier_scores)
        mean_score = np.mean(scores)
        
        # 1. 筛选出所有得分高于平均值的 "高价值" 边界点
        high_score_indices = np.where(scores > mean_score)[0]
        
        if len(high_score_indices) < 2:
            # 如果高价值点少于2个，TSP无意义，直接选择最高分或退化
            if len(high_score_indices) == 1:
                return self.frontier_locations[high_score_indices[0]]
            else:
                return self._explore_geometric()

        high_score_frontiers = self.frontier_locations[high_score_indices]
        
        N = len(high_score_frontiers)
        
        # 2. 构建成本矩阵 (N+1) x (N+1)，包含智能体当前位置
        cost_matrix = np.full((N + 1, N + 1), 1e6) # 初始化为大数值
        
        # 点集：[智能体位置, 高分边界点1, 高分边界点2, ...]
        points = [(self.start[0] + 1, self.start[1] + 1)] + \
                 [(f[0] + 1, f[1] + 1) for f in high_score_frontiers]
        
        planner = FMMPlanner(self.traversible)

        # 3. 计算点与点之间的导航距离
        for i in range(N + 1):
            start_point = points[i]
            # 计算从 start_point 到所有其他点的FMM距离
            planner.set_goal(start_point)
            fmm_dist = planner.fmm_dist[::-1]

            for j in range(N + 1):
                if i == j:
                    cost_matrix[i, j] = 0
                    continue

                end_point = points[j]
                
                # 确保坐标在fmm_dist的有效范围内
                if 0 <= end_point[0] < fmm_dist.shape[0] and 0 <= end_point[1] < fmm_dist.shape[1]:
                    dist = fmm_dist[end_point[0], end_point[1]]
                    if not (np.isinf(dist) or np.isnan(dist)):
                        cost_matrix[i, j] = dist
        
        # 4. 解TSP问题
        try:
            # 使用动态规划求解器
            permutation, _ = solve_tsp_dynamic_programming(cost_matrix)
        except Exception as e:
            print(f"TSP solver failed: {e}. Falling back to geometric exploration.")
            return self._explore_geometric()

        if not permutation or len(permutation) < 2:
            return self._explore_geometric()

        # 5. 选择最优路径的第一个边界点作为目标
        next_point_index_in_tsp_list = permutation[1]
        
        # TSP列表中的索引要-1才能对应到high_score_frontiers中的索引
        next_frontier_idx_in_high_score_list = next_point_index_in_tsp_list - 1
        
        goal = high_score_frontiers[next_frontier_idx_in_high_score_list]
        
        return goal

    # ===================================================================
    # >>> 结束：新增自适应探索所需的核心辅助函数 <<<
    # ===================================================================
    # ===================================================================
    # >>> 核心改造封装：新增自适应探索策略函数 <<<
    # ===================================================================

    def _adaptive_explore(self):
        """
        封装了 ApexNav 思想的自适应探索策略，作为 UniGoal Stage 1 的实现。
        该策略会根据环境语义线索的强弱，动态选择探索模式。

        返回:
            np.ndarray: 计算出的下一个目标点坐标 (goal)。
        """
        print("UniGoal Stage 1 (Zero Matching): Entering adaptive exploration.")
        
        # 确保我们有一个目标子图
        if not hasattr(self, 'subgraph') or self.subgraph is None:
             if self.goalgraph_decomposed and 'subgraph_1' in self.goalgraph_decomposed:
                 self.subgraph = self.goalgraph_decomposed['subgraph_1']
             else:
                 # 如果没有分解的子图，使用整个目标图
                 self.subgraph = self.goalgraph

        # 1. 计算前沿点的语义得分
        frontier_scores = self._calculate_frontier_semantic_scores()
        
        # 2. 如果没有有效的前沿或得分，则退回到UniGoal原始的探索方法
        if not frontier_scores:
            print("Adaptive Strategy: No valid frontiers or scores. Falling back to default UniGoal exploration.")
            # 调用原有的UniGoal explore_subgraph逻辑
            return self.explore_subgraph()

        # 3. 计算分布指标
        s_max = np.max(frontier_scores) if frontier_scores else 0
        s_mean = np.mean(frontier_scores) if frontier_scores else 0
        s_std = np.std(frontier_scores) if frontier_scores else 0
        
        # 确保s_mean不为零，避免除以零
        ratio = s_max / s_mean if s_mean > 0 else 0

        # 4. 从配置中读取阈值
        r_threshold = getattr(self.args, 'adaptive_r_threshold', 1.10)
        sigma_threshold = getattr(self.args, 'adaptive_sigma_threshold', 0.015)

        # 5. 根据指标进行决策
        if ratio > r_threshold and s_std > sigma_threshold:
            # 语义线索强 -> 执行基于TSP的语义探索
            print(f"Adaptive Strategy: Strong semantic cues (Ratio: {ratio:.2f} > {r_threshold}, Std: {s_std:.3f} > {sigma_threshold}). Using TSP semantic exploration.")
            goal = self._explore_semantic_tsp(frontier_scores)
        else:
            # 语义线索弱 -> 执行基于几何的探索
            print(f"Adaptive Strategy: Weak semantic cues (Ratio: {ratio:.2f}, Std: {s_std:.3f}). Using geometric exploration.")
            goal = self._explore_geometric()
            
        return goal


    def _update_semantic_score_map(self):
        """
        使用VLM评估当前观测，并更新全局的语义得分图 (semantic_score_map)。
        此函数完全遵循 ApexNav (Sec. IV-A2) 的思想。
        (已修正，可以正确处理字符串格式的 self.subgraph)
        """
        # 1. 获取最新的观测和智能体位姿
        if not hasattr(self, 'image_rgb') or not hasattr(self, 'image_depth') or self.image_rgb is None or self.image_depth is None:
            return
            
        rgb_image_pil = Image.fromarray(self.image_rgb)
        depth_tensor = torch.from_numpy(self.image_depth).to(self.device)

        # 2. 获取目标描述 (*** 此处是关键修改 ***)
        if not hasattr(self, 'subgraph') or self.subgraph is None:
            return

        target_nodes = []
        # 检查 self.subgraph 的类型
        if isinstance(self.subgraph, str):
            # 如果是字符串，我们进行解析
            try:
                # 找到 "Nodes:" 和 "Edges:" 之间的部分
                nodes_part = self.subgraph.split("Nodes:")[1].split("Edges:")[0]
                # 去除首尾空格和句号，然后按逗号分割
                target_nodes = [node.strip() for node in nodes_part.strip().strip('.').split(',') if node.strip()]
            except IndexError:
                # 如果解析失败（例如格式不符），我们就无法确定目标
                print(f"Warning: Could not parse target nodes from subgraph string: '{self.subgraph}'")
                return
                
        elif isinstance(self.subgraph, dict) and 'nodes' in self.subgraph:
            # 如果是字典，则使用旧的逻辑
            target_nodes = [node['id'] for node in self.subgraph['nodes']]

        # 如果没有解析出任何有效的目标节点，则退出
        if not target_nodes:
            return
        
        target_object_text = " or ".join(target_nodes)
        print(f"Semantic Map Update: VLM is looking for '{target_object_text}'")

        # 3. 构建VLM Prompt (这部分不变)
        prompt = f"Based on the image, does it seem like there is a '{target_object_text}' ahead? Answer with a confidence score from 0.0 (no) to 1.0 (yes). Respond with only the numerical score."

        # 4. 调用VLM获取当前帧的语义得分 (这部分不变)
        try:
            vlm_score_str = self.vlm(prompt, rgb_image_pil)
            current_score = float(vlm_score_str.strip())
            current_score = max(0.0, min(1.0, current_score))
        except (ValueError, TypeError, AttributeError) as e:
            vlm_score_str = "N/A"
            print(f"VLM Error: Could not parse score from '{vlm_score_str}'. Error: {e}. Defaulting to 0.5.")
            current_score = 0.5

        # 5. 将得分投影到地图上 (这部分逻辑不变，但依赖于下一节添加的 get_visible_points 函数)

        input_pose = np.zeros(7)
        if not hasattr(self, 'full_pose') or self.full_pose is None:
            return

        # Correctly access the first row of the tensor before converting to numpy
        pose_numpy = self.full_pose[0].cpu().numpy()
        input_pose[:3] = pose_numpy
        input_pose[1] = self.map_size_cm / 100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        
        _, start = self.get_traversible(self.full_map.cpu().numpy()[0, 0, ::-1], input_pose)

        # Correctly access the yaw value from the first row (index 0), third column (index 2)
        agent_yaw = self.full_pose[0, 2].item()
        
        if not hasattr(self, 'get_visible_points'):
            print("Semantic Map Update: 'get_visible_points' method not found. Cannot project scores.")
            return

        visible_points_map_coords = self.get_visible_points(depth_tensor.cpu().numpy(), start, agent_yaw)
        if visible_points_map_coords is None:
            return

        for (mx, my) in visible_points_map_coords:
            current_confidence = 0.8
            if not (0 <= mx < self.semantic_score_map.shape[2] and 0 <= my < self.semantic_score_map.shape[3]):
                continue
                
            prev_score = self.semantic_score_map[0, 0, mx, my]
            prev_confidence = self.semantic_confidence_map[0, 0, mx, my]

            new_confidence = prev_confidence + current_confidence
            
            if new_confidence > 0:
                new_score = ((prev_score * prev_confidence) + (current_score * current_confidence)) / new_confidence
            else:
                new_score = 0
            
            self.semantic_score_map[0, 0, mx, my] = new_score
            self.semantic_confidence_map[0, 0, mx, my] = new_confidence



    def get_visible_points(self, depth, start, agent_yaw):
        """
        根据深度图和智能体位姿，计算在2D地图上所有可见的自由空间点。
        """
        # *** 关键修复：确保深度图是二维的 ***
        # 检查depth数组的维度
        if depth.ndim == 3 and depth.shape[2] == 1:
            # 如果是 (H, W, 1) 的形状, 就把它压缩成 (H, W)
            depth = np.squeeze(depth, axis=2)
        
        # 深度图尺寸
        # 现在 depth 保证是二维的，所以这里不会出错
        H, W = depth.shape
        
        # 创建像素坐标网格
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        
        # 过滤掉无效的深度值 (例如，大于最大感知范围)
        # 假设配置中有最大深度, 如果没有则使用一个默认值
        max_depth = getattr(self.args, 'max_depth', 30.0) 
        valid_depth_mask = (depth > 0) & (depth < max_depth)
        
        # 仅处理有效的像素
        # 现在 valid_depth_mask 是 (H, W)，与 xs, ys, depth 的维度匹配
        xs, ys, ds = xs[valid_depth_mask], ys[valid_depth_mask], depth[valid_depth_mask]
        
        # 如果没有有效的可见点，直接返回
        if len(xs) == 0:
            return None

        # 从像素坐标和深度计算相机坐标系中的3D点
        # K是相机内参矩阵
        if not hasattr(self, 'camera_matrix'):
            return None # 缺少相机参数则无法计算
        K_inv = np.linalg.inv(self.camera_matrix)
        
        # (u, v, 1) * d
        cam_coords = K_inv @ np.vstack((xs * ds, ys * ds, ds))

        # 将点从相机坐标系转换到世界坐标系
        rot_cam_to_agent = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        rot_agent_to_world = np.array([[np.cos(agent_yaw), -np.sin(agent_yaw), 0],
                                    [np.sin(agent_yaw),  np.cos(agent_yaw), 0],
                                    [0, 0, 1]])

        world_points = (rot_agent_to_world @ rot_cam_to_agent @ cam_coords).T

        # 将3D世界坐标投影到2D地图坐标
        map_coords_x = (self.map_size / 2) + (world_points[:, 0] * 100 / self.map_resolution)
        map_coords_y = (self.map_size / 2) - (world_points[:, 1] * 100 / self.map_resolution)
        
        map_coords = np.vstack((map_coords_y, map_coords_x)).T.astype(int)
        
        # 过滤掉地图外的点
        valid_map_mask = (map_coords[:, 0] >= 0) & (map_coords[:, 0] < self.map_size) & \
                        (map_coords[:, 1] >= 0) & (map_coords[:, 1] < self.map_size)
        
        map_coords = map_coords[valid_map_mask]
        
        # 去重
        if len(map_coords) > 0:
            return np.unique(map_coords, axis=0)
        else:
            return None
        # ===================================================================
    # >>> 全新功能：语义地图可视化 <<<
    # ===================================================================
    def visualize_semantic_map(self):
        """
        将当前的语义得分图渲染成一个彩色的热力图，并与障碍物地图叠加，
        最后保存为图像文件，以便于调试和分析。
        """
        # 1. 从PyTorch Tensor中提取数据并转换为NumPy数组
        # 我们只关心得分图，置信度图仅用于计算
        if not hasattr(self, 'semantic_score_map') or self.semantic_score_map is None:
            print("Semantic Map Visualization: No semantic_score_map available. Skipping visualization.")
            return
        
        score_map_np = self.semantic_score_map[0, 0].cpu().numpy()
        
        # 2. 将得分 (0.0 - 1.0) 归一化到 (0 - 255) 以便应用色彩映射
        # 我们乘以255并转换为8位无符号整数
        heatmap_gray = (score_map_np * 255).astype(np.uint8)
        
        # 3. 应用色彩映射（例如JET），将灰度图变为彩色热力图
        # 低分区域为蓝色，高分区域为红色
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        
        # 4. 创建一个背景：使用障碍物地图
        # self.full_map[0, 0] 是障碍物地图
        obstacle_map_np = self.full_map[0, 0].cpu().numpy()
        # 将其转换为三通道的BGR图像，以便与彩色热力图融合
        # 障碍物显示为灰色
        background = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        background[obstacle_map_np == 1] = [128, 128, 128] # 灰色障碍物
        background[obstacle_map_np == 0] = [255, 255, 255] # 白色自由空间

        # 5. 将热力图与背景融合
        # 我们只在有得分的区域（非黑色区域）显示热力图
        mask = heatmap_gray > 0
        # 使用加权融合，让背景和热力图都能看到
        # alpha是热力图权重，beta是背景权重
        fused_map = cv2.addWeighted(heatmap_color, 0.7, background, 0.3, 0)
        # 只在有得分的区域应用融合效果
        background[mask] = fused_map[mask]

        # 6. 在地图上绘制智能体的当前位置
        # 我们需要从 full_pose 获取2D地图坐标
        # 注意：full_pose 的 y 坐标需要从世界坐标系翻转到图像坐标系
        # We need from full_pose to get the 2D map coordinates
        # Correctly access x from the first row (index 0), first column (index 0)
        agent_x = int(self.full_pose[0, 0].item())
        # Correctly access y from the first row (index 0), second column (index 1)
        agent_y = int(self.full_pose[0, 1].item())
        
        # 绘制一个红色的圆圈代表智能体
        cv2.circle(background, (agent_x, agent_y), radius=5, color=(0, 0, 255), thickness=-1)

        # 7. 保存图像
        # 创建一个专门用于存放可视化结果的文件夹
        output_dir = "debug_visuals"
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用导航步数命名文件，以便观察变化
        # 假设 self.navigate_steps 存在
        step_count = self.navigate_steps
        output_path = os.path.join(output_dir, f"semantic_map_step_{step_count:04d}.png")
        
        cv2.imwrite(output_path, background)
        print(f"Semantic map visualization saved to '{output_path}'")