import glob
import os
import cv2
import torch
import numpy as np


def _normalize_img(img_tensor: np.ndarray, min_val=-1, max_val=1) -> np.ndarray:
    """
    将 [-1,1] 或其他范围的图像数据归一化到 [0,255] 并转换为 uint8。
    img_tensor: shape [..., H, W] 或 [..., H, W, C]
    """
    # 如果原本就是 [0,255]，可根据实际需求直接返回
    # 这里只是示例写法，可根据情况做不同处理
    img_tensor = (img_tensor - min_val) / (max_val - min_val + 1e-8)
    img_tensor = np.clip(img_tensor, 0, 1) * 255
    return img_tensor.astype(np.uint8)


def _depth_to_gray(depth_tensor: np.ndarray, depth_min=0, depth_max=80) -> np.ndarray:
    """
    将深度 [0,80] 映射到 [0,255] 的灰度图
    depth_tensor: shape [H, W] 或 [T, H, W]
    """
    depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-8)
    depth_tensor = np.clip(depth_tensor, 0, 1) * 255
    depth_tensor = depth_tensor.astype(np.uint8)
    return depth_tensor


def _to_3channels(img_tensor: np.ndarray) -> np.ndarray:
    """
    将单通道数据堆成 3 通道；若已是 3 通道，则原样返回。
    img_tensor: shape [H, W] 或 [H, W, C]
    """
    if len(img_tensor.shape) == 2:
        # [H, W]
        return np.stack([img_tensor] * 3, axis=-1)
    elif img_tensor.shape[-1] == 1:
        # [H, W, 1]
        return np.concatenate([img_tensor] * 3, axis=-1)
    else:
        # 认为已经是 3 通道 [H, W, 3]
        return img_tensor


def save_videos_set(
    first_frames: torch.Tensor,  # [B, 512, 512, 3], 值域 [0,255]
    depths: torch.Tensor,  # [B, 512, 512], 值域 [0,80]
    mask: torch.Tensor,  # [B, 49, 512, 512], 值域 {0,1}
    mask_latent: torch.Tensor,  # [B, 49, 512, 512], 值域 [-1,1]
    mask_warped: torch.Tensor,  # [B, 49, 3, 512, 512], 值域 [-1,1]
    mask_pixel_values: torch.Tensor,  # [B, 49, 3, 512, 512], 值域 [-1,1]
    pixel_values: torch.Tensor,  # [B, 49, 3, 512, 512], 值域 [-1,1]
    save_path: str,  # 存储视频的文件夹路径，输出视频会以下标自动命名
):
    """
    将同一个 batch（batch_size=B）的数据合成为 B 个视频文件，
    在单个视频内将 6 个“子画面”拼接成 2 行 × 3 列的形式。

    Args:
        first_frames: torch.Size([B, 512, 512, 3]), 取值范围为 [0,255]
        depths: torch.Size([B, 512, 512]), 取值范围 [0,80]
        mask: torch.Size([B, 49, 512, 512]), 取值范围为 {0,1}
        mask_latent: torch.Size([B, 49, 512, 512]), 取值范围 [-1,1]
        warped: torch.Size([B, 49, 3, 512, 512]), 取值范围 [0,255]
        mask_pixel_values: torch.Size([B, 49, 3, 512, 512]), 取值范围 [-1,1]
        pixel_values: torch.Size([B, 49, 3, 512, 512]), 取值范围 [-1,1]
        save_path: 视频保存文件夹路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 转成 numpy 并移动到 CPU 上（若已经在 CPU 则无需 .cpu()）
    first_frames_np = first_frames.cpu().numpy()
    depths_np = depths.detach().to(torch.float32).cpu().numpy()
    mask_np = mask.cpu().numpy()
    warped_np = mask_warped.detach().to(torch.float32).cpu().numpy()
    mask_pixel_values_np = mask_pixel_values.to(torch.float32).cpu().numpy()
    pixel_values_np = pixel_values.to(torch.float32).cpu().numpy()

    B = first_frames_np.shape[0]  # batch_size
    _, T, _, _ = mask_np.shape  # 这里假设 mask.shape = [B, 49, 512, 512] => T=49

    # 获取当前目录中已经存在的以 "video_" 开头、".mp4" 结尾的文件数量
    existing_files = glob.glob(os.path.join(save_path, "video_*.mp4"))
    start_index = len(existing_files)  # 从已有文件数量继续往后编号

    # 视频写入相关设置
    fps = 14
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i in range(B):
        # 1) 准备 6 路“子视频”各自的帧序列

        # (a) first_frames[i] (仅 1 帧) -> 复制 T 次
        #     shape为 (512, 512, 3), 值域 [0,255]
        first_frame_array = np.tile(first_frames_np[i][None, ...], (T, 1, 1, 1))
        # shape: [T, 512, 512, 3]

        # (b) depths[i] (仅 1 帧) -> 复制 T 次，并转为灰度 3 通道
        depth_gray = _depth_to_gray(depths_np[i])  # [512,512] -> [0,255]
        depth_map_array = np.tile(depth_gray[None, ...], (T, 1, 1))  # [T,512,512]
        depth_map_array = np.stack([_to_3channels(x) for x in depth_map_array], axis=0)
        # shape: [T,512,512,3]

        # (c) mask[i] -> shape [49,512,512], 值域 {0,1}
        #     转为 [T,512,512], 再乘 255，变 3 通道
        mask_i = mask_np[i]  # [49,512,512]
        mask_i = (mask_i * 255).astype(np.uint8)  # [T,512,512]
        mask_array = np.stack([_to_3channels(x) for x in mask_i], axis=0)
        # shape: [T,512,512,3]

        # (d) mask_latent[i] -> shape [49,512,512], 值域 [-1,1]
        #     转为 [T,512,512], 再乘 255，变 3 通道
        mask_latent_i = mask_latent[i]  # [49,512,512]
        mask_latent_i = (mask_i * 255).astype(np.uint8)  # [T,512,512]
        mask_latent_array = np.stack([_to_3channels(x) for x in mask_latent_i], axis=0)
        # shape: [T,512,512,3]

        # (e) warped[i] -> shape [49,3,512,512], 值域 [-1,1]
        #     先归一化到 [0,255], 再转 [T,512,512,3]
        warped_i = warped_np[i]  # [49,3,512,512]
        warped_i = _normalize_img(warped_i, -1, 1)
        warped_i = np.transpose(warped_i, (0, 2, 3, 1))  # [T,512,512,3]

        # (f) mask_pixel_values[i] -> shape [49,3,512,512], 值域 [-1,1]
        #     先归一化到 [0,255], 再转 [T,512,512,3]
        mpv_i = mask_pixel_values_np[i]  # [49,3,512,512]
        mpv_i = _normalize_img(mpv_i, -1, 1)
        mpv_i = np.transpose(mpv_i, (0, 2, 3, 1))  # [T,512,512,3]

        # (g) pixel_values[i] -> 同样处理
        pv_i = pixel_values_np[i]  # [49,3,512,512]
        pv_i = _normalize_img(pv_i, -1, 1)
        pv_i = np.transpose(pv_i, (0, 2, 3, 1))  # [T,512,512,3]

        # (h) 纯黑色背景
        black_bg = np.zeros((512, 512, 3), dtype=np.uint8)
        black_bg = np.tile(black_bg[None, ...], (T, 1, 1, 1))
        # shape: [T, 512, 512, 3]

        # 2) 创建 VideoWriter
        #   单帧拼接后分辨率： 高度 = 2*512, 宽度 = 3*512
        out_h = 2 * 512
        out_w = 4 * 512

        video_index = start_index + i
        video_fn = os.path.join(save_path, f"video_{video_index}.mp4")
        out_writer = cv2.VideoWriter(video_fn, fourcc, fps, (out_w, out_h))

        # 3) 拼帧并写入视频
        for t in range(T):
            # row1: [first_frame, depth_map, mask]
            row1 = np.concatenate([first_frame_array[t], depth_map_array[t], mask_array[t], mask_latent_array[t]], axis=1)
            # row2: [warped, mask_pixel_values, pixel_values]
            row2 = np.concatenate([warped_i[t], mpv_i[t], pv_i[t], black_bg], axis=1)
            # 拼成 2 行
            frame = np.concatenate([row1, row2], axis=0)  # shape: (1024, 1536, 3)

            # 写入视频
            # 如果前面是 RGB 排列，需要将其转为 BGR
            frame_bgr = frame[..., ::-1].astype(np.uint8)
            out_writer.write(frame_bgr)

        out_writer.release()

    print(f"所有视频已保存在: {save_path}")
