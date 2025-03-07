import glob
import os
import cv2
from einops import rearrange
import torch
import numpy as np
import matplotlib.cm
from EasyCamera.warp import get_mask_batch
import torch.nn.functional as F
from torchvision import transforms
from easyanimate.pipeline.pipeline_easyanimate_inpaint import add_noise_to_reference_video


def _normalize_img(img_tensor: np.ndarray, min_val=-1, max_val=1) -> np.ndarray:
    """
    将 [-1,1] 或其他范围的图像数据归一化到 [0,255] 并转换为 uint8。
    img_tensor: shape [..., H, W] 或 [..., H, W, C]
    """
    # 如果原本就是 [0,255]，可根据实际需求直接返回
    # 这里只是示例写法，可根据情况做不同处理
    if min_val == -1 and max_val == 1:
        img_tensor = img_tensor * 0.5 + 0.5
    else:
        img_tensor = (img_tensor - min_val) / (max_val - min_val + 1e-8)
    img_tensor = np.clip(img_tensor, 0, 1) * 255
    return img_tensor.astype(np.uint8)


def colorize(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """
    将单帧或多帧深度图（值数组）映射为颜色图：
      - 当输入为二维数组时，shape 为 [H, W]；
      - 当输入为三维数组时，shape 为 [F, H, W]，其中 F 表示帧数。
    """
    # 如果传入的是 torch.Tensor，则转换到 numpy 数组，并 squeeze 去除冗余维度
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)

    # 判断是否为多帧数据
    if value.ndim == 3:  # 多帧深度图，shape 为 [F, H, W]
        colored_frames = []
        for i in range(value.shape[0]):
            frame = value[i]
            # 如果没有提供 invalid_mask，则根据 invalid_val 构造掩膜
            if invalid_mask is None:
                inv_mask = frame == invalid_val
            else:
                # 支持 invalid_mask 为 [F, H, W] 或者单帧 [H, W]（对多帧均适用）
                if invalid_mask.ndim == 3:
                    inv_mask = invalid_mask[i]
                else:
                    inv_mask = invalid_mask
            mask = np.logical_not(inv_mask)

            # 如果未传入 vmin/vmax，则根据有效像素计算当前帧的归一化范围
            vmin_frame = np.percentile(frame[mask], 2) if vmin is None else vmin
            vmax_frame = np.percentile(frame[mask], 85) if vmax is None else vmax

            if vmin_frame != vmax_frame:
                norm_frame = (frame - vmin_frame) / (vmax_frame - vmin_frame)
            else:
                norm_frame = frame * 0.0

            # 将无效像素设为 NaN
            norm_frame[inv_mask] = np.nan

            if value_transform:
                norm_frame = value_transform(norm_frame)

            # 生成当前帧的 RGBA 颜色图，shape 为 [H, W, 4]
            colored_frame = cmapper(norm_frame, bytes=True)
            colored_frame[inv_mask] = background_color

            if gamma_corrected:
                colored_frame = colored_frame / 255.0
                colored_frame = np.power(colored_frame, 2.2)
                colored_frame = colored_frame * 255

            colored_frames.append(colored_frame.astype(np.uint8))

        # 堆叠所有帧，得到 shape 为 [F, H, W, 4]
        colored = np.stack(colored_frames, axis=0)
        return colored

    elif value.ndim == 2:  # 单帧深度图，shape 为 [H, W]
        if invalid_mask is None:
            invalid_mask = value == invalid_val
        mask = np.logical_not(invalid_mask)

        vmin_calc = np.percentile(value[mask], 2) if vmin is None else vmin
        vmax_calc = np.percentile(value[mask], 85) if vmax is None else vmax

        if vmin_calc != vmax_calc:
            norm_value = (value - vmin_calc) / (vmax_calc - vmin_calc)
        else:
            norm_value = value * 0.0

        norm_value[invalid_mask] = np.nan

        if value_transform:
            norm_value = value_transform(norm_value)

        colored = cmapper(norm_value, bytes=True)
        colored[invalid_mask] = background_color

        if gamma_corrected:
            colored = colored / 255.0
            colored = np.power(colored, 2.2)
            colored = colored * 255

        return colored.astype(np.uint8)
    else:
        raise ValueError("输入数组应为二维 (H, W) 或三维 (F, H, W)！")


def _depth_to_gray(depth_tensor: np.ndarray, depth_min=0, depth_max=20, cmap='magma_r') -> np.ndarray:
    """
    将深度图映射为彩色图。

    原本该函数仅做归一化和灰度映射，这里改为调用 colorize()，使用指定的 colormap，生成 RGB 彩色图。

    Args:
        depth_tensor (np.ndarray): 深度图，形状 F, 512, 512，值域一般为 [depth_min, depth_max]。
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
    first_frames: torch.Tensor,  # [B, 49, 3, 512, 512], 值域 [-1,1]
    depths: torch.Tensor,  # [B, F, 512, 512], 值域 [0,80]
    mask: torch.Tensor,  # [B, 49, 512, 512], 值域 {0,1}
    mask_warped: torch.Tensor,  # [B, 49, 3, 512, 512], 值域 [-1,1]
    mask_pixel_values: torch.Tensor,  # [B, 49, 3, 512, 512], 值域 [-1,1] torch.Size([1, 3, 49, 512, 512])
    pixel_values: torch.Tensor,  # [B, 49, 3, 512, 512], 值域 [-1,1]
    save_path: str,  # 存储视频的文件夹路径，输出视频会以下标自动命名
    fps: int = 14,
):
    """
    将同一个 batch（batch_size=B）的数据合成为 B 个视频文件，
    在单个视频内将 6 个“子画面”拼接成 2 行 × 3 列的形式。

    Args:
        first_frames: torch.Size([B, 49, 3, 512, 512]), 取值范围为 [-1,1]
        depths: torch.Size([B, F, 512, 512]), 取值范围 [0,80]
        mask: torch.Size([B, 49, 512, 512]), 取值范围为 {0,1}
        warped: torch.Size([B, 49, 3, 512, 512]), 取值范围 [0,255]
        mask_pixel_values: torch.Size([B, 49, 3, 512, 512]), 取值范围 [-1,1]
        pixel_values: torch.Size([B, 49, 3, 512, 512]), 取值范围 [-1,1]
        save_path: 视频保存文件夹路径
    """
    if save_path.endswith(".mp4"):
        dir_path = os.path.dirname(save_path)
    else:
        dir_path = save_path
    os.makedirs(dir_path, exist_ok=True)

    # 转成 numpy 并移动到 CPU 上（若已经在 CPU 则无需 .cpu()）
    first_frames_np = first_frames.to(torch.float32).cpu().numpy()
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i in range(B):
        # 1) 准备 6 路“子视频”各自的帧序列

        # (a) first_frames[i] (仅 1 帧) -> 复制 T 次
        #     shape为 (512, 512, 3), 值域 [0,255]
        ff_i = first_frames_np[i]  # [49, 512, 512, 3]
        ff_i = _normalize_img(ff_i, -1, 1)
        ff_i = np.transpose(ff_i, (0, 2, 3, 1))  # [T,512,512,3]
        # shape: [T, 512, 512, 3]

        # (b) depths[i] (仅 1 帧) -> 复制 T 次，并转为灰度 3 通道
        # depth_gray = _depth_to_gray(depths_np[i])  # [512,512] -> [0,255]
        depth_map_array = _depth_to_gray(depths_np[i])  # [F, 512, 512] -> [T, 512, 512, 3]
        # depth_map_array = np.tile(depth_color[None, ...], (T, 1, 1, 1))  # [T,512,512,3]
        # depth_map_array = np.stack([_to_3channels(x) for x in depth_map_array], axis=0)
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

        if save_path.endswith(".mp4"):
            video_fn = save_path
        else:
            video_index = start_index + i
            video_fn = os.path.join(save_path, f"video_{video_index}.mp4")
        out_writer = cv2.VideoWriter(video_fn, fourcc, fps, (out_w, out_h))

        # 3) 拼帧并写入视频
        for t in range(T):
            # row1: [first_frame, depth_map, mask]
            row1 = np.concatenate([ff_i[t], depth_map_array[t], mask_array[t]], axis=1)
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


def prepare_depth_anything(dav2_model, dav2_outdoor):
    from extern.DepthAnythingV2.metric_depth.depth_anything_v2.dpt import (
        DepthAnythingV2,
    )

    dav2_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # Depth Anything V2
    dav2_model_config = {
        **dav2_model_configs[dav2_model],
        # 20 for indoor model, 80 for outdoor model
        'max_depth': 80 if dav2_outdoor else 20,
    }
    depth_anything = DepthAnythingV2(**dav2_model_config)

    # Change the path to the
    dav2_model_fn = f'depth_anything_v2_metric_{"vkitti" if dav2_outdoor else "hypersim"}_{dav2_model}.pth'
    depth_anything.load_state_dict(torch.load(f'./models/checkpoints_dav2/{dav2_model_fn}', map_location='cpu'))

    return depth_anything


def pre_process_first_frames(first_frames, device, dtype, input_size=518):
    # 1. 记录原图大小
    first_frames = first_frames * 0.5 + 0.5
    b, f, c, h, w = first_frames.shape
    original_hw = (h, w)

    # # 2. 将像素值从 0–255 转为 0–1，并转为 float
    # frames = first_frames.float() / 255.0

    # 3. 通道从 [N, H, W, C] -> [N, C, H, W]
    # frames = frames.permute(0, 3, 1, 2)  # [batch_size, 3, 512, 512]
    frames = rearrange(first_frames, "b f c h w -> (b f) c h w")

    # 4. 缩放到指定大小 (可根据需要调整或去掉)
    frames = F.interpolate(frames, size=(input_size, input_size), mode='bicubic', align_corners=False)

    # 5. 按照给定的 mean 和 std 进行归一化
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    frames = rearrange(frames, "(b f) c h w -> b f c h w", f=f)

    # 6. 放到指定的计算设备上
    frames = frames.to(dtype=dtype, device=device)

    return frames, original_hw


def resize_mask(
    mask: torch.Tensor, latents_shape, process_first_frame_only: bool = True, mode: str = "trilinear", align_corners: bool = False  # (B, C, F, H, W)  # (B, C', F', H', W')
):
    """
    尝试复刻您原先的逻辑:
      1) 如果 process_first_frame_only=True，则只把 mask 的第 0 帧插值到目标 F' 维度=1 的大小，
         其它帧插值到 F' - 1，然后拼起来。
      2) 否则一次性插值到 (F', H', W')。

    注意: 这里假设 mask.shape = (B, C, F, H, W)；latents.shape = (B, C', F', H', W')。
    """
    b, c, f, h, w = mask.shape
    _, _, f2, h2, w2 = latents_shape

    if process_first_frame_only:
        # 目标尺寸先复制一下
        target_size = [f2, h2, w2]

        # 先处理第 0 帧 => 目标 F 方向固定为 1
        target_size_first = [1, h2, w2]
        mask_first = mask[:, :, 0:1, :, :]  # (B, C, 1, H, W)
        resized_first = F.interpolate(mask_first.float(), size=target_size_first, mode=mode, align_corners=align_corners)  # (B, C, 1, H2, W2)

        # 接着处理剩余帧 => F 维度 = f2 - 1
        if f2 > 1:
            target_size_rest = [f2 - 1, h2, w2]
            mask_rest = mask[:, :, 1:, :, :]  # (B, C, F-1, H, W)
            resized_rest = F.interpolate(mask_rest.float(), size=target_size_rest, mode=mode, align_corners=align_corners)  # (B, C, f2 - 1, H2, W2)

            # 拼起来 => (B, C, f2, H2, W2)
            resized_mask = torch.cat([resized_first, resized_rest], dim=2)
        else:
            resized_mask = resized_first
    else:
        # 一次性插值到 (f2, h2, w2)
        resized_mask = F.interpolate(mask.float(), size=(f2, h2, w2), mode=mode, align_corners=align_corners)

    return resized_mask


# def add_noise_to_reference_video(image: torch.Tensor, ratio: float = None, mean: float = -3.0, std: float = 0.5) -> torch.Tensor:
#     """
#     给 reference video（形如 (B, ...) 的张量）添加噪声。

#     参数：
#     • image:           形状 (B, [...])，例如 (B, F, C, H, W)。默认会对 batch 维度逐个随机生成噪声强度。
#     • ratio:           如果为 None，则使用 lognormal 分布生成 sigma；否则将使用固定的 ratio 作为 sigma。
#     • mean, std:       当 ratio 为 None 时，sigma 的 log空间服从的正态分布 N(mean, std) 的参数。

#     返回：
#     • 加完噪声后的图像，形状与输入相同。对于 image == -1 的位置，不添加噪声（噪声置 0）。
#     """
#     device = image.device
#     dtype = image.dtype
#     batch_size = image.shape[0]

#     # 1. 如果 ratio=None，则随机生成一个 lognormal 分布的噪声强度 sigma (B,)
#     if ratio is None:
#         # 生成服从 N(mean, std) 的随机数，然后取 exp => lognormal
#         sigma = torch.normal(mean=mean, std=std, size=(batch_size,), device=device, dtype=dtype)
#         sigma = sigma.exp()  # (B,)
#     else:
#         # ratio 不是 None，直接当作固定噪声强度
#         # 假设 ratio 是一个标量；若需要支持 ratio 为张量，可在此做更多检查与广播
#         sigma = torch.full((batch_size,), ratio, device=device, dtype=dtype)

#     # 2. 生成与 image 同形状的噪声，再乘以各自的 sigma
#     #    sigma.view(batch_size, 1, 1, 1, 1) 保证可以广播到 image 的所有维度 (B,F,C,H,W)
#     #    如果您有其他维度格式，需要自行修改这里的广播
#     shape_ones = [1] * (image.ndim - 1)  # 减去 batch 维度后剩余的维度数量
#     image_noise = torch.randn_like(image) * sigma.view(batch_size, *shape_ones)

#     # 3. 对于 image == -1 的位置不加噪声 => 即把噪声置 0
#     image_noise = torch.where(image == -1, torch.zeros_like(image_noise), image_noise)

#     # 4. 最终结果
#     return image + image_noise


def get_warped_from_depth(
    depths: torch.Tensor,  # torch.Size([1, 49, 512, 512])
    first_frames: torch.Tensor,  # torch.Size([1, 49, 3, 512, 512])
    camera_poses: torch.Tensor,  # (B, F, 19)
    ori_hs: torch.Tensor,  # (B,)
    ori_ws: torch.Tensor,  # (B,)
    video_sample_size: int,  # 512
    pixel_values: torch.Tensor,  # (B, F, 3, 512, 512)
    weight_dtype,
    accelerator_device,
    latents_shape: list,  # (B, 16, 13, 64, 64)
    vae_cache_mag_vae,
):
    mask, warped = get_mask_batch(first_frames, depths, camera_poses, ori_hs, ori_ws, video_sample_size)  # => torch.Size([1, 49, 512, 512]), torch.Size([1, 49, 3, 512, 512])
    mask_for_pixel = mask.unsqueeze(2)  # torch.Size([1, 49, 1, 512, 512])
    mask_pixel_values = pixel_values * (mask_for_pixel < 0.5) + torch.ones_like(pixel_values) * (mask_for_pixel > 0.5) * -1  # torch.Size([1, 49, 3, 512, 512])

    video_transforms = transforms.Compose(
        [
            transforms.Resize(video_sample_size),
            transforms.CenterCrop((video_sample_size, video_sample_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False),
        ]
    )

    warped = warped / 255.0
    warped = video_transforms(warped)  # torch.Size([1, 49, 3, 512, 512])

    # mask_warped = warped * (1 - mask_for_pixel) + (-1) * mask_for_pixel
    mask_warped = warped * (mask_for_pixel < 0.5) + torch.ones_like(warped) * (mask_for_pixel > 0.5) * -1
    mask_warped = mask_warped.to(accelerator_device, dtype=weight_dtype)  # torch.Size([1, 49, 3, 512, 512])

    mask_all_ones = (mask == 1).reshape(mask.shape[0], -1).all(dim=1)  # (B,)
    rand_values = torch.rand_like(mask_all_ones.float())
    # 当 mask_all_ones=True 且 rand<0.9 => flag=0，否则=1
    t2v_flag = torch.where(
        (mask_all_ones & (rand_values < 0.9)),
        torch.zeros_like(mask_all_ones, dtype=mask_all_ones.dtype),
        torch.ones_like(mask_all_ones, dtype=mask_all_ones.dtype),
    )
    # 转到指定设备
    t2v_flag = t2v_flag.to(accelerator_device, dtype=weight_dtype)

    # 4. 编码 mask，本质上就是 1-mask 并缩放到与 latents 同尺寸
    #    先把 mask 从 (B, F, H, W) -> (B, 1, F, H, W)
    mask_reshape = rearrange(mask, "b f h w -> b 1 f h w")  # torch.Size([1, 1, 49, 512, 512])
    mask_reshape = 1 - mask_reshape
    mask_reshape = resize_mask(mask_reshape, latents_shape, vae_cache_mag_vae)  # torch.Size([1, 1, 13, 64, 64])

    # 5. 可选：对原视频添加噪声
    mask_pixel_values_with_noise = add_noise_to_reference_video(mask_warped)  # torch.Size([1, 49, 3, 512, 512])
    # mask_pixel_values_with_noise = mask_warped  # torch.Size([1, 49, 3, 512, 512])

    return t2v_flag, mask_pixel_values_with_noise, mask, mask_reshape, mask_pixel_values, mask_warped
