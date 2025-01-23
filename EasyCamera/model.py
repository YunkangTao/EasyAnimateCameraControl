import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from EasyCamera.warp import get_mask_batch
from einops import rearrange
import torchvision.transforms as transforms


def _batch_encode_vae(pixel_values, vae, vae_mini_batch, weight_dtype, video_length):
    """
    将原先的内嵌函数独立出来，做小幅优化。
    """
    # pixel_values: (B, F, 3, 512, 512) or (B, F, C, H, W)
    # VAE 可能有 5 维的卷积权重（3D VAE），也可能是普通 4 维 ...
    if vae.quant_conv is None or (vae.quant_conv.weight is not None and vae.quant_conv.weight.ndim == 5):
        # 3D VAE 的情况
        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
        bs = vae_mini_batch
        outputs = []
        for i in range(0, pixel_values.shape[0], bs):
            pv_bs = pixel_values[i : i + bs]
            latent_dist = vae.encode(pv_bs)[0]  # (bs, c, f, h, w)
            latent_sample = latent_dist.sample()
            outputs.append(latent_sample)
        latents = torch.cat(outputs, dim=0)
    else:
        # 常规 2D VAE 的情况
        # 先合并 B 和 F 为 BF
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        bs = vae_mini_batch
        outputs = []
        for i in range(0, pixel_values.shape[0], bs):
            pv_bs = pixel_values[i : i + bs].to(dtype=weight_dtype)
            latent_dist = vae.encode(pv_bs).latent_dist
            latent_sample = latent_dist.sample()
            outputs.append(latent_sample)
        latents = torch.cat(outputs, dim=0)  # (B*F, C', H', W')
        # 再恢复到 (B, C', F, H', W')
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    return latents


def resize_mask(
    mask: torch.Tensor, latents: torch.Tensor, process_first_frame_only: bool = True, mode: str = "trilinear", align_corners: bool = False  # (B, C, F, H, W)  # (B, C', F', H', W')
):
    """
    尝试复刻您原先的逻辑:
      1) 如果 process_first_frame_only=True，则只把 mask 的第 0 帧插值到目标 F' 维度=1 的大小，
         其它帧插值到 F' - 1，然后拼起来。
      2) 否则一次性插值到 (F', H', W')。

    注意: 这里假设 mask.shape = (B, C, F, H, W)；latents.shape = (B, C', F', H', W')。
    """
    b, c, f, h, w = mask.shape
    _, _, f2, h2, w2 = latents.shape

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


def add_noise_to_reference_video(image: torch.Tensor, ratio: float = None, mean: float = -3.0, std: float = 0.5) -> torch.Tensor:
    """
    给 reference video（形如 (B, ...) 的张量）添加噪声。

    参数：
    • image:           形状 (B, [...])，例如 (B, F, C, H, W)。默认会对 batch 维度逐个随机生成噪声强度。
    • ratio:           如果为 None，则使用 lognormal 分布生成 sigma；否则将使用固定的 ratio 作为 sigma。
    • mean, std:       当 ratio 为 None 时，sigma 的 log空间服从的正态分布 N(mean, std) 的参数。

    返回：
    • 加完噪声后的图像，形状与输入相同。对于 image == -1 的位置，不添加噪声（噪声置 0）。
    """
    device = image.device
    dtype = image.dtype
    batch_size = image.shape[0]

    # 1. 如果 ratio=None，则随机生成一个 lognormal 分布的噪声强度 sigma (B,)
    if ratio is None:
        # 生成服从 N(mean, std) 的随机数，然后取 exp => lognormal
        sigma = torch.normal(mean=mean, std=std, size=(batch_size,), device=device, dtype=dtype)
        sigma = sigma.exp()  # (B,)
    else:
        # ratio 不是 None，直接当作固定噪声强度
        # 假设 ratio 是一个标量；若需要支持 ratio 为张量，可在此做更多检查与广播
        sigma = torch.full((batch_size,), ratio, device=device, dtype=dtype)

    # 2. 生成与 image 同形状的噪声，再乘以各自的 sigma
    #    sigma.view(batch_size, 1, 1, 1, 1) 保证可以广播到 image 的所有维度 (B,F,C,H,W)
    #    如果您有其他维度格式，需要自行修改这里的广播
    shape_ones = [1] * (image.ndim - 1)  # 减去 batch 维度后剩余的维度数量
    image_noise = torch.randn_like(image) * sigma.view(batch_size, *shape_ones)

    # 3. 对于 image == -1 的位置不加噪声 => 即把噪声置 0
    image_noise = torch.where(image == -1, torch.zeros_like(image_noise), image_noise)

    # 4. 最终结果
    return image + image_noise


def get_inpaint_latents_from_depth(
    depths: torch.Tensor,  # (B, 512, 512)
    first_frames: torch.Tensor,  # (B, 512, 512, 3)
    camera_poses: torch.Tensor,  # (B, F, 19)
    ori_hs: torch.Tensor,  # (B,)
    ori_ws: torch.Tensor,  # (B,)
    video_sample_size: int,  # 512
    pixel_values: torch.Tensor,  # (B, F, 3, 512, 512)
    weight_dtype,
    accelerator_device,
    latents: torch.Tensor,  # (B, 16, 13, 64, 64)
    vae,
    vae_mini_batch: int,  # 1
    video_length: int,  # 49
):
    """
    新的函数去除了对 batch_size 的显式循环，并对部分操作做了矢量化提升。
    """
    # 1. 批量计算 mask: (B, F, H, W)
    mask, warped = get_mask_batch(first_frames, depths, camera_poses, ori_hs, ori_ws, video_sample_size)  # => (B, F, 512, 512)

    # 注意 pixel_values 形状为 (B, F, 3, 512, 512)
    # 与 mask (B, F, 512, 512) 广播时，需要在通道维度上做扩展
    mask_for_pixel = mask.unsqueeze(2)  # (B, F, 1, 512, 512)

    # 2. 生成 inpaint 像素：被 mask 的位置替换为 -1，其它使用原值
    mask_pixel_values = pixel_values * (1 - mask_for_pixel) + (-1) * mask_for_pixel

    video_transforms = transforms.Compose(
        [
            transforms.Resize(video_sample_size),
            transforms.CenterCrop((video_sample_size, video_sample_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    warped = warped / 255.0
    warped = video_transforms(warped)

    mask_warped = warped * (1 - mask_for_pixel) + (-1) * mask_for_pixel
    mask_warped = mask_warped.to(accelerator_device, dtype=weight_dtype)

    # 3. 处理 t2v_flag 的逻辑
    # 原代码:
    #     t2v_flag = [(_mask == 1).all() for _mask in mask]
    #     for _mask in t2v_flag:
    #         if _mask and np.random.rand() < 0.90: ...
    #
    # 这里矢量化：
    #    先判断整段 mask 是否全部为 1
    #    rand < 0.9 时则置 0，否则置 1
    #
    # mask => (B, F, H, W)
    # (mask == 1).all(dim=(1,2,3)) => (B,) 表示该 batch 下所有像素都为 1 的布尔值
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
    mask_reshape = rearrange(mask, "b f h w -> b 1 f h w")
    mask_reshape = 1 - mask_reshape
    mask_reshape = resize_mask(mask_reshape, latents, vae.cache_mag_vae)

    # 5. 可选：对原视频添加噪声
    mask_pixel_values_with_noise = add_noise_to_reference_video(mask_warped)

    # 6. 编码 inpaint latents：将 mask_pixel_values 传进 VAE 得到 mask_latents
    mask_latents = _batch_encode_vae(mask_pixel_values_with_noise, vae, vae_mini_batch, weight_dtype, video_length)

    # 拼接 inpaint_latents: (B, 1+潜编码通道, F, H, W)
    inpaint_latents = torch.cat([mask_reshape, mask_latents], dim=1)

    # 与 t2v_flag 相乘
    # t2v_flag shape: (B,) => 需要扩维到 (B, 1, 1, 1, 1)
    inpaint_latents = inpaint_latents * t2v_flag.view(-1, 1, 1, 1, 1)

    # 按照 VAE scaling_factor 进行缩放
    inpaint_latents = inpaint_latents * vae.config.scaling_factor

    return inpaint_latents, mask_pixel_values, mask, mask_warped


def pre_process_first_frames(first_frames, device, dtype, input_size=518):
    # 1. 记录原图大小
    batch_size, h, w, c = first_frames.shape
    original_hw = (h, w)

    # 2. 将像素值从 0–255 转为 0–1，并转为 float
    frames = first_frames.float() / 255.0

    # 3. 通道从 [N, H, W, C] -> [N, C, H, W]
    frames = frames.permute(0, 3, 1, 2)  # [batch_size, 3, 512, 512]

    # 4. 缩放到指定大小 (可根据需要调整或去掉)
    frames = F.interpolate(frames, size=(input_size, input_size), mode='bicubic', align_corners=False)

    # 5. 按照给定的 mean 和 std 进行归一化
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # 6. 放到指定的计算设备上
    frames = frames.to(dtype=dtype, device=device)

    return frames, original_hw


class EasyCamera(nn.Module):
    def __init__(self, easyanimate, depth_anything_v2):
        super().__init__()
        self.easyanimate = easyanimate
        self.depth_anything_v2 = depth_anything_v2

    def forward(
        self,
        first_frames,
        camera_poses,
        ori_hs,
        ori_ws,
        video_sample_size,
        pixel_values,
        weight_dtype,
        accelerator_device,
        latents,
        vae,
        vae_mini_batch,
        video_length,
        noisy_latents,
        timesteps,
        prompt_embeds,
        prompt_attention_mask,
        prompt_embeds_2,
        prompt_attention_mask_2,
        add_time_ids,
        style,
        image_rotary_emb,
        clip_encoder_hidden_states,
        clip_attention_mask,
    ):
        first_frames_processed, (h, w) = pre_process_first_frames(first_frames, accelerator_device, weight_dtype)  # 2,3,518,518
        depths = self.depth_anything_v2.forward(first_frames_processed)  # torch.Size([2, 518, 518])
        depths = F.interpolate(depths.unsqueeze(1), (h, w), mode="bilinear", align_corners=True).squeeze(1)  # torch.Size([2, 512, 512])

        inpaint_latents, mask_pixel_values, mask, mask_warped = get_inpaint_latents_from_depth(
            depths,
            first_frames,
            camera_poses,
            ori_hs,
            ori_ws,
            video_sample_size,
            pixel_values,
            weight_dtype,
            accelerator_device,
            latents,
            vae,
            vae_mini_batch,
            video_length,
        )

        del vae
        torch.cuda.empty_cache()

        inpaint_latents = inpaint_latents.to(noisy_latents.dtype)
        noise_pred = self.easyanimate(
            noisy_latents,
            timesteps.to(noisy_latents.dtype),
            encoder_hidden_states=prompt_embeds,
            text_embedding_mask=prompt_attention_mask,
            encoder_hidden_states_t5=prompt_embeds_2,
            text_embedding_mask_t5=prompt_attention_mask_2,
            image_meta_size=add_time_ids,
            style=style,
            image_rotary_emb=image_rotary_emb,
            inpaint_latents=inpaint_latents,
            clip_encoder_hidden_states=clip_encoder_hidden_states,
            clip_attention_mask=clip_attention_mask,
            return_dict=False,
        )[0]

        return (noise_pred, first_frames, depths, mask, mask_warped, mask_pixel_values, pixel_values)
