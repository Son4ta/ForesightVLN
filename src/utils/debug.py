import os
import cv2
import numpy as np


def save_debug_images(rgb_image, depth_image, timestep=129, episode_id=129, save_dir="debug_output"):
    """
    保存 RGB 和深度图以便调试。

    Args:
        rgb_image (np.ndarray): RGB 图像数组 (H, W, 3)，uint8。
        depth_image (np.ndarray): 深度图数组 (H, W, 1)，float。
        timestep (int): 当前的时间步。
        episode_id (int): 当前的 episode 编号。
        save_dir (str): 保存图像的根目录。
    """
    # 如果两张图都不存在，就没必要创建文件夹了
    if rgb_image is None and depth_image is None:
        return

    # 1. 创建保存路径
    episode_save_path = os.path.join(save_dir, f"episode_{episode_id:03d}")
    os.makedirs(episode_save_path, exist_ok=True)

    # 2. 检查并保存 RGB 图像
    if rgb_image is not None:
        try:
            # OpenCV 使用 BGR 格式，obs['rgb'] 是 RGB, 需要转换
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(episode_save_path, f"rgb_{timestep:04d}.png"), rgb_bgr)
        except Exception as e:
            print(f"Warning: Failed to save RGB image at step {timestep}. Error: {e}")

    # 3. 检查并保存深度图
    if depth_image is not None:
        try:
            # 移除单通道维度 (H, W, 1) -> (H, W)
            depth_visual = depth_image.squeeze()
            # 归一化并应用伪彩色
            depth_visual = cv2.normalize(depth_visual, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(episode_save_path, f"depth_{timestep:04d}.png"), depth_visual)
        except Exception as e:
            print(f"Warning: Failed to save Depth image at step {timestep}. Error: {e}")
