import glob
import os
import cv2
import torch
import numpy as np
import matplotlib.cm


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


def colorize(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor or np.ndarray): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W).
        vmin (float, optional): vmin-valued entries are mapped to the start color of cmap.
                                If None, value.min() is used (通过百分位数计算). Defaults to None.
        vmax (float, optional): vmax-valued entries are mapped to the end color of cmap.
                                If None, value.max() is used (通过百分位数计算). Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): 指定无效的像素值，这部分像素会上色为背景色。 Defaults to -99.
        invalid_mask (np.ndarray, optional): 无效区域的布尔掩模。 Defaults to None.
        background_color (tuple[int], optional): 无效像素的背景颜色（RGBA）。 Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): 是否对结果进行 gamma 校正。 Defaults to False.
        value_transform (Callable, optional): 对有效像素进行变换的函数。 Defaults to None.

    Returns:
        np.ndarray: 颜色化的深度图，数据类型为 uint8，形状为 (H, W, 4)。
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # 根据有效像素计算归一化范围（也可以使用显式传入的 vmin/vmax）
    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax

    if vmin != vmax:
        norm_value = (value - vmin) / (vmax - vmin)
    else:
        norm_value = value * 0.0

    # 对无效值设置为 NaN
    norm_value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        norm_value = value_transform(norm_value)
    colored = cmapper(norm_value, bytes=True)  # 得到 (H, W, 4) 的 RGBA 图像
    colored[invalid_mask] = background_color

    if gamma_corrected:
        colored = colored / 255.0
        colored = np.power(colored, 2.2)
        colored = colored * 255
    return colored.astype(np.uint8)


def _depth_to_gray(depth_tensor: np.ndarray, depth_min=0, depth_max=80, cmap='magma_r') -> np.ndarray:
    """
    将深度图映射为彩色图。

    原本该函数仅做归一化和灰度映射，这里改为调用 colorize()，使用指定的 colormap，生成 RGB 彩色图。

    Args:
        depth_tensor (np.ndarray): 深度图，形状 (H, W)，值域一般为 [depth_min, depth_max]。
        depth_min (float, optional): 最小深度值。 Defaults to 0.
        depth_max (float, optional): 最大深度值。 Defaults to 80.
        cmap (str, optional): 使用的 matplotlib colormap。 Defaults to 'magma_r'.

    Returns:
        np.ndarray: 颜色化后的深度图，形状 (H, W, 3)，类型为 uint8。
    """
    # 限定深度范围
    depth_tensor = np.clip(depth_tensor, depth_min, depth_max)
    # 使用 colorize 生成颜色映射，显式传入 vmin 和 vmax
    colored_depth = colorize(depth_tensor, vmin=depth_min, vmax=depth_max, cmap=cmap)
    # colorize 返回 RGBA (4 通道)，这里只保留 RGB 三通道
    colored_depth = colored_depth[..., :3]
    return colored_depth


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
        # depth_gray = _depth_to_gray(depths_np[i])  # [512,512] -> [0,255]
        depth_color = _depth_to_gray(depths_np[i])  # [512,512,3]
        depth_map_array = np.tile(depth_color[None, ...], (T, 1, 1, 1))  # [T,512,512,3]
        depth_map_array = np.stack([_to_3channels(x) for x in depth_map_array], axis=0)
        # shape: [T,512,512,3]

        # (c) mask[i] -> shape [49,512,512], 值域 {0,1}
        #     转为 [T,512,512], 再乘 255，变 3 通道
        mask_i = mask_np[i]  # [49,512,512]
        mask_i = (mask_i * 255).astype(np.uint8)  # [T,512,512]
        mask_array = np.stack([_to_3channels(x) for x in mask_i], axis=0)
        # shape: [T,512,512,3]

        # (d) warped[i] -> shape [49,3,512,512], 值域 [-1,1]
        #     先归一化到 [0,255], 再转 [T,512,512,3]
        warped_i = warped_np[i]  # [49,3,512,512]
        warped_i = _normalize_img(warped_i, -1, 1)
        warped_i = np.transpose(warped_i, (0, 2, 3, 1))  # [T,512,512,3]

        # (e) mask_pixel_values[i] -> shape [49,3,512,512], 值域 [-1,1]
        #     先归一化到 [0,255], 再转 [T,512,512,3]
        mpv_i = mask_pixel_values_np[i]  # [49,3,512,512]
        mpv_i = _normalize_img(mpv_i, -1, 1)
        mpv_i = np.transpose(mpv_i, (0, 2, 3, 1))  # [T,512,512,3]

        # (f) pixel_values[i] -> 同样处理
        pv_i = pixel_values_np[i]  # [49,3,512,512]
        pv_i = _normalize_img(pv_i, -1, 1)
        pv_i = np.transpose(pv_i, (0, 2, 3, 1))  # [T,512,512,3]

        # 2) 创建 VideoWriter
        #   单帧拼接后分辨率： 高度 = 2*512, 宽度 = 3*512
        out_h = 2 * 512
        out_w = 3 * 512

        video_index = start_index + i
        video_fn = os.path.join(save_path, f"video_{video_index}.mp4")
        out_writer = cv2.VideoWriter(video_fn, fourcc, fps, (out_w, out_h))

        # 3) 拼帧并写入视频
        for t in range(T):
            # row1: [first_frame, depth_map, mask]
            row1 = np.concatenate([first_frame_array[t], depth_map_array[t], mask_array[t]], axis=1)
            # row2: [warped, mask_pixel_values, pixel_values]
            row2 = np.concatenate([warped_i[t], mpv_i[t], pv_i[t]], axis=1)
            # 拼成 2 行
            frame = np.concatenate([row1, row2], axis=0)  # shape: (1024, 1536, 3)

            # 写入视频
            # 如果前面是 RGB 排列，需要将其转为 BGR
            frame_bgr = frame[..., ::-1].astype(np.uint8)
            out_writer.write(frame_bgr)

        out_writer.release()

    print(f"所有视频已保存在: {save_path}")
