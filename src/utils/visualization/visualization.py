import colorsys
from PIL import Image, ImageDraw, ImageFont
import cv2
import skimage
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    # >>> FIX: Explicitly set dtype to np.int32 for OpenCV compatibility <<<
    return np.array([pt1, pt2, pt3, pt4], dtype=np.int32)


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image(goal_name, args):
    vis_image = np.ones((655-100, 1380-200-25, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    # text = f"{goal_name}"
    # textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    # textX = (215 - textsize[0]) // 2 + 25
    # textY = (50 + textsize[1]) // 2
    # vis_image = cv2.putText(vis_image, text, (textX, textY),
    #                         font, fontScale, color, thickness,
    #                         cv2.LINE_AA)

    if args.goal_type == 'ins-image ':
        text = f"Goal Image"
    elif args.goal_type == 'text':
        text = f"Goal Text"
    elif args.goal_type == 'object':
        text = f"Goal Object"
    else:
        text = f"Goal"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (215 - textsize[0]) // 2 + 25
    if args.environment == 'habitat' or args.environment == 'ai2thor':
        textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    if args.environment == 'habitat' or args.environment == 'ai2thor':
        text = f"Goal Graph"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (215 - textsize[0]) // 2 + 25
        textY = (50 + textsize[1]) // 2 + 265
        vis_image = cv2.putText(vis_image, text, (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)

    text = "Observation RGB"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 215 + (360 - textsize[0]) // 2 + 40
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)
    
    text = "Occupancy Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 840 + (480 - textsize[0]) // 2 - 190
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Curiousity Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 1320 + (360 - textsize[0]) // 2 + 60
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)
    
    return vis_image


def line_list(text, line_length=22):
    text_list = []
    for i in range(0, len(text), line_length):
        text_list.append(text[i:(i + line_length)])
    return text_list


def add_text_list(image: np.ndarray, text_list: list, position=(10, 20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(0, 0, 0), thickness=1, highlight_line_index=[]):
    highlight_color = (0, 0, 0)
    not_highlight_color = (128, 128, 128)
    for i, text in enumerate(text_list):
        position_i = (position[0], position[1] + i * 15)
        color = highlight_color if len(highlight_line_index) == 0 or i in highlight_line_index else not_highlight_color
        cv2.putText(image, text, position_i, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

'''
新增↓
'''

def init_vis_image_v2(goal_name, args):
    """
    >>> v3.1 LAYOUT <<<
    调整右下角两个面板的比例。
    """
    width, height = 1800, 900
    vis_image = np.ones((height, width, 3), dtype=np.uint8) * 245
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    color = (0, 0, 0)
    thickness = 1

    # --- 定义面板位置和尺寸 ---
    margin = 30
    col1_width = 800
    col2_width = width - col1_width - 3 * margin
    col2_x_start = col1_width + 2 * margin

    # --- Panel 1: 主视角 ---
    p1_x, p1_y, p1_w, p1_h = margin, margin, col1_width, height - 2 * margin
    cv2.putText(vis_image, "Agent's Egocentric View", (p1_x, p1_y - 10), font, font_scale, color, thickness)
    cv2.rectangle(vis_image, (p1_x, p1_y), (p1_x + p1_w, p1_y + p1_h), (180, 180, 180), 1)

    # --- 右侧列 ---
    # --- Panel 2: 全局地图 ---
    p2_h = int((height - 3 * margin) * 0.6)
    p2_x, p2_y, p2_w = col2_x_start, margin, col2_width
    cv2.putText(vis_image, "Global Map & Planning", (p2_x, p2_y - 10), font, font_scale, color, thickness)
    cv2.rectangle(vis_image, (p2_x, p2_y), (p2_x + p2_w, p2_y + p2_h), (180, 180, 180), 1)

    p3_y_start = p2_y + p2_h + margin
    p3_h = height - p3_y_start - margin

    # --- FIX: 调整面板宽度比例 (Task Goal 35%, Semantic Map 65%) ---
    p3_w = int((col2_width - margin) * 0.35)
    p3_x = col2_x_start
    cv2.putText(vis_image, "Task Goal", (p3_x, p3_y_start - 10), font, font_scale, color, thickness)
    cv2.rectangle(vis_image, (p3_x, p3_y_start), (p3_x + p3_w, p3_y_start + p3_h), (180, 180, 180), 1)

    p4_x_start = p3_x + p3_w + margin
    p4_w = col2_width - p3_w - margin
    p4_x = p4_x_start
    cv2.putText(vis_image, "Semantic Value Map", (p4_x, p3_y_start - 10), font, font_scale, color, thickness)
    cv2.rectangle(vis_image, (p4_x, p3_y_start), (p4_x + p4_w, p3_y_start + p3_h), (180, 180, 180), 1)

    return vis_image

def draw_frontiers(map_img, frontiers, color=(0, 255, 255), radius=3):
    """在地图上绘制前沿点。"""
    if frontiers is None or len(frontiers) == 0:
        return map_img
    
    for (y, x) in frontiers:
        # 注意OpenCV的坐标顺序是(x, y)
        cv2.circle(map_img, (int(x), int(y)), radius, color, -1)
    return map_img

def draw_tsp_path(map_img, frontiers, path, color=(255, 0, 255), thickness=2):
    """在地图上绘制TSP路径。"""
    if frontiers is None or path is None or len(path) < 2:
        return map_img

    # 将路径索引映射到实际坐标
    path_coords = frontiers[path]

    # 绘制路径线段
    for i in range(len(path_coords) - 1):
        pt1 = (int(path_coords[i][1]), int(path_coords[i][0]))       # (x, y)
        pt2 = (int(path_coords[i+1][1]), int(path_coords[i+1][0])) # (x, y)
        cv2.line(map_img, pt1, pt2, color, thickness)
    
    return map_img