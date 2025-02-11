from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
from splatting import cpu as splatting_cpu
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

if torch.cuda.is_available():
    from splatting import cuda as splatting_cuda
else:
    splatting_cuda = None


def get_projection_matrix(
    fovy: torch.Tensor,  # shape (B,)
    aspect_wh: float,
    near: float,
    far: float,
) -> torch.Tensor:
    """
    返回形状 (B, 4, 4) 的投影矩阵。
    """
    batch_size = fovy.shape[0]
    # 直接先计算 1 / tan(fovy/2)
    f = 1.0 / torch.tan(fovy * 0.5)
    # 初始化
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=fovy.dtype, device=fovy.device)
    proj_mtx[:, 0, 0] = f / aspect_wh
    proj_mtx[:, 1, 1] = -f  # y 轴翻转
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_src_proj_mtx(focal_length_x_norm, focal_length_y_norm, height, width, res, src_image):
    """
    根据相机内参和图像处理步骤计算投影矩阵。

    参数:
    - focal_length_x_norm (float): 归一化的x方向焦距 (fx / width)
    - focal_length_y_norm (float): 归一化的y方向焦距 (fy / height)
    - height (int): 原始图像高度
    - width (int): 原始图像宽度
    - res (int): 图像缩放后的尺寸 (res, res)
    - src_image (torch.Tensor): 源图像张量，用于确定设备类型

    返回:
    - src_proj_mtx (torch.Tensor): 投影矩阵，形状为 (1, 4, 4)
    """
    # 将归一化焦距转换为像素单位
    focal_length_x = focal_length_x_norm * width
    focal_length_y = focal_length_y_norm * height

    # 裁剪为中心正方形
    cropped_size = min(width, height)
    scale_crop_x = cropped_size / width
    scale_crop_y = cropped_size / height

    # 调整焦距以适应裁剪后的图像
    focal_length_x_cropped = focal_length_x * scale_crop_x
    focal_length_y_cropped = focal_length_y * scale_crop_y

    # 缩放图像
    scale_resize = res / cropped_size
    focal_length_x_resized = focal_length_x_cropped * scale_resize
    focal_length_y_resized = focal_length_y_cropped * scale_resize

    # 计算垂直视场角 (fovy) 使用调整后的焦距和缩放后的高度
    fovy = 2.0 * torch.atan(torch.tensor(res / (2.0 * focal_length_y_resized)))
    fovy = fovy.unsqueeze(0)  # 形状调整为 (1,)

    near, far = 0.01, 100.0
    aspect_wh = 1.0  # 因为图像被缩放为正方形 (res, res)

    # 获取投影矩阵
    src_proj_mtx = get_projection_matrix(fovy=fovy, aspect_wh=aspect_wh, near=near, far=far).to(src_image)

    return src_proj_mtx


def get_src_proj_mtx_batch(
    focal_length_x_norm: torch.Tensor,  # (B,)
    focal_length_y_norm: torch.Tensor,  # (B,)
    height: torch.Tensor,  # (B,)
    width: torch.Tensor,  # (B,)
    res: int,
    src_image: torch.Tensor,  # (B, 3, H, W) 用于推断 device
) -> torch.Tensor:
    """
    批量计算投影矩阵。
    """
    device = src_image.device
    dtype = src_image.dtype
    B = focal_length_x_norm.shape[0]

    # 1. 转换为像素单位
    focal_length_x = focal_length_x_norm * width
    focal_length_y = focal_length_y_norm * height

    # 2. 裁剪得到中心正方形
    cropped_size = torch.min(width, height).float()  # (B,)
    scale_crop_x = cropped_size / width
    scale_crop_y = cropped_size / height

    # 焦距裁剪
    fx_cropped = focal_length_x * scale_crop_x
    fy_cropped = focal_length_y * scale_crop_y

    # 3. 图像再缩放
    scale_resize = res / cropped_size  # (B,)
    fx_resized = fx_cropped * scale_resize
    fy_resized = fy_cropped * scale_resize

    # 4. 计算 fovy
    # fovy = 2 * arctan(res / (2 * fy_resized))
    fovy = 2.0 * torch.atan(res / (2.0 * fy_resized + 1e-8))
    fovy = fovy.to(device=device, dtype=dtype)  # (B,)

    near, far = 0.01, 100.0
    aspect_wh = 1.0  # 正方形

    # 调用 get_projection_matrix 得到 (B, 4, 4)
    src_proj_mtx = get_projection_matrix(fovy, aspect_wh, near, far)
    return src_proj_mtx.to(device=device, dtype=dtype)


def convert_camera_extrinsics(w2c: torch.Tensor) -> torch.Tensor:
    """
    批量化版本：若 w2c 形状为 (..., 3, 4)，就对其做相同处理。
    x、y 翻转，z 不变。
    """
    device = w2c.device
    dtype = w2c.dtype

    # S: (3,3)
    S = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=dtype))

    # w2c: (..., 3, 4)
    # 拆分 R (3x3), t(3x1)
    R = w2c[..., :3]  # (..., 3, 3)
    t = w2c[..., 3]  # (..., 3)

    new_R = S @ R
    new_t = S @ t.unsqueeze(-1)  # (..., 3, 1)

    # 组装新的外参
    # new_w2c: (..., 3, 4)
    new_w2c = torch.cat([new_R, new_t], dim=-1)
    return new_w2c


def get_rel_view_mtx(src_wc, tar_wc, src_image):
    src_wc = convert_camera_extrinsics(src_wc)
    tar_wc = convert_camera_extrinsics(tar_wc)

    # 将第一个 W2C 矩阵扩展为 4x4 齐次变换矩阵
    T1 = torch.eye(4, dtype=src_wc.dtype, device=src_wc.device)
    T1[:3, :3] = src_wc[:, :3]
    T1[:3, 3] = src_wc[:, 3]

    # 将第二个 W2C 矩阵扩展为 4x4 齐次变换矩阵
    T2 = torch.eye(4, dtype=tar_wc.dtype, device=tar_wc.device)
    T2[:3, :3] = tar_wc[:, :3]
    T2[:3, 3] = tar_wc[:, 3]

    # 计算第一个视图矩阵的逆
    T1_inv = torch.inverse(T1)

    # 计算相对视图矩阵
    rel_view_mtx = T2 @ T1_inv

    return rel_view_mtx.to(src_image)


def get_rel_view_mtx_batch(
    src_wc: torch.Tensor,  # (B, 3, 4)
    tar_wc: torch.Tensor,  # (B, F, 3, 4)
) -> torch.Tensor:
    """
    批量计算相对视图变换矩阵:
    rel_view_mtx = T2 @ inv(T1).

    返回形状: (B, F, 4, 4).
    """
    device = src_wc.device
    dtype = src_wc.dtype

    B, F = tar_wc.shape[0], tar_wc.shape[1]

    # 1. 转换成 4x4 齐次矩阵
    #    src_wc => (B, 3, 4) => (B, 4, 4)
    src_wc_4x4 = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)
    src_wc_4x4[:, :3, :4] = convert_camera_extrinsics(src_wc)

    # 2. tar_wc => (B, F, 3, 4) => (B, F, 4, 4)
    tar_wc_4x4 = torch.eye(4, dtype=dtype, device=device).repeat(B, F, 1, 1)
    tar_wc_extr = convert_camera_extrinsics(tar_wc.view(B * F, 3, 4))
    tar_wc_4x4 = tar_wc_4x4.view(B * F, 4, 4)  # 先展平
    tar_wc_4x4[:, :3, :4] = tar_wc_extr
    tar_wc_4x4 = tar_wc_4x4.view(B, F, 4, 4)

    # 3. 求逆: T1_inv => (B, 4, 4)
    T1_inv = torch.inverse(src_wc_4x4)

    # 4. rel_view_mtx => (B, F, 4, 4)
    #    需要将 T1_inv (B,4,4) 扩展到 (B,F,4,4) 后与 tar_wc_4x4 做矩阵乘法
    T1_inv_expanded = T1_inv.unsqueeze(1).expand(-1, F, -1, -1)  # (B, F, 4, 4)
    rel_view_mtx = tar_wc_4x4 @ T1_inv_expanded
    return rel_view_mtx


def get_viewport_matrix(width: int, height: int, batch_size: int = 1, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    返回形状 (B, 4, 4) 的 viewport matrix。
    """
    N = (
        torch.tensor(
            [
                [width / 2, 0, 0, width / 2],
                [0, height / 2, 0, height / 2],
                [0, 0, 1 / 2, 1 / 2],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
            device=device,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    return N


def preprocess_image(image: torch.Tensor, size=512) -> torch.Tensor:
    return F.interpolate(image, (size, size))


class Embedder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed


class SummationSplattingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame, flow):
        assert frame.dtype == flow.dtype
        assert frame.device == flow.device
        assert len(frame.shape) == 4
        assert len(flow.shape) == 4
        assert frame.shape[0] == flow.shape[0]
        assert frame.shape[2] == flow.shape[2]
        assert frame.shape[3] == flow.shape[3]
        assert flow.shape[1] == 2
        ctx.save_for_backward(frame, flow)
        output = torch.zeros_like(frame)
        if frame.is_cuda:
            if splatting_cuda is not None:
                splatting_cuda.splatting_forward_cuda(frame, flow, output)
            else:
                raise RuntimeError("splatting.cuda is not available")
        else:
            splatting_cpu.splatting_forward_cpu(frame, flow, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        frame, flow = ctx.saved_tensors
        grad_frame = torch.zeros_like(frame)
        grad_flow = torch.zeros_like(flow)
        if frame.is_cuda:
            if splatting_cuda is not None:
                splatting_cuda.splatting_backward_cuda(frame, flow, grad_output, grad_frame, grad_flow)
            else:
                raise RuntimeError("splatting.cuda is not available")
        else:
            splatting_cpu.splatting_backward_cpu(frame, flow, grad_output, grad_frame, grad_flow)
        return grad_frame, grad_flow


SPLATTING_TYPES = ["summation", "average", "linear", "softmax"]


def splatting_function(
    splatting_type: str,
    frame: torch.Tensor,
    flow: torch.Tensor,
    importance_metric: Union[torch.Tensor, None] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    if splatting_type == "summation":
        assert importance_metric is None
    elif splatting_type == "average":
        assert importance_metric is None
        importance_metric = frame.new_ones([frame.shape[0], 1, frame.shape[2], frame.shape[3]])
        frame = torch.cat([frame, importance_metric], 1)
    elif splatting_type == "linear":
        assert isinstance(importance_metric, torch.Tensor)
        assert importance_metric.shape[0] == frame.shape[0]
        assert importance_metric.shape[1] == 1
        assert importance_metric.shape[2] == frame.shape[2]
        assert importance_metric.shape[3] == frame.shape[3]
        frame = torch.cat([frame * importance_metric, importance_metric], 1)
    elif splatting_type == "softmax":
        assert isinstance(importance_metric, torch.Tensor)
        assert importance_metric.shape[0] == frame.shape[0]
        assert importance_metric.shape[1] == 1
        assert importance_metric.shape[2] == frame.shape[2]
        assert importance_metric.shape[3] == frame.shape[3]

        # # 检查是否存在 NaN 或 Inf
        # if torch.isnan(importance_metric).any() or torch.isinf(importance_metric).any():
        #     raise ValueError("importance_metric contains NaN or Inf before exp()")

        # # 检查最大最小值
        # max_val = importance_metric.max()
        # min_val = importance_metric.min()
        # print(f"Before exp - max: {max_val}, min: {min_val}")

        importance_metric = importance_metric.exp()
        frame = torch.cat([frame * importance_metric, importance_metric], 1)
    else:
        raise NotImplementedError("splatting_type has to be one of {}, not '{}'".format(SPLATTING_TYPES, splatting_type))

    output = SummationSplattingFunction.apply(frame, flow)

    if splatting_type != "summation":
        output = output[:, :-1, :, :] / (output[:, -1:, :, :] + eps)

    return output


def forward_warper(
    image: Float[Tensor, 'B C H W'],
    screen: Float[Tensor, 'B (H W) 2'],
    pcd: Float[Tensor, 'B (H W) 4'],
    mvp_mtx: Float[Tensor, 'B 4 4'],
    viewport_mtx: Float[Tensor, 'B 4 4'],
    alpha: float = 0.5,
) -> Dict[str, Tensor]:
    H, W = image.shape[2:4]

    # Projection.
    points_c = pcd @ mvp_mtx.mT
    points_ndc = points_c / (points_c[..., 3:4] + 1e-8)
    # To screen.
    coords_new = points_ndc @ viewport_mtx.mT

    # Masking invalid pixels.
    invalid = coords_new[..., 2] <= 0
    coords_new[invalid] = -1000000 if coords_new.dtype == torch.float32 else -1e4

    # Calculate flow and importance for splatting.
    new_z = points_c[..., 2:3]
    flow = coords_new[..., :2] - screen[..., :2]
    ## Importance.
    importance = alpha / (new_z + 1e-8)
    importance = importance - importance.amin((1, 2), keepdim=True)
    importance = importance / importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10
    ## Rearrange.
    importance = rearrange(importance, 'b (h w) c -> b c h w', h=H, w=W)
    flow = rearrange(flow, 'b (h w) c -> b c h w', h=H, w=W)

    # Splatting.
    warped = splatting_function('softmax', image, flow, importance, eps=1e-6)
    ## mask is 1 where there is no splat
    mask = (warped == 0.0).all(dim=1, keepdim=True).to(image.dtype)
    flow2 = rearrange(coords_new[..., :2], 'b (h w) c -> b c h w', h=H, w=W)

    output = dict(warped=warped, mask=mask, correspondence=flow2)

    return output


def warp_function(
    src_image: Float[Tensor, 'B C H W'],
    src_depth: Float[Tensor, 'B C H W'],
    rel_view_mtx: Float[Tensor, 'B 4 4'],
    src_proj_mtx: Float[Tensor, 'B 4 4'],
    tar_proj_mtx: Float[Tensor, 'B 4 4'],
    # viewport_mtx: Float[Tensor, 'B 4 4'],
):
    device = "cuda"
    dtype = torch.float16

    batch_size = src_image.shape[0]

    viewport_mtx: Float[Tensor, 'B 4 4'] = get_viewport_matrix(512, 512, batch_size=1, device=device).to(dtype)

    # Rearrange and resize.
    src_image = preprocess_image(src_image)
    src_depth = preprocess_image(src_depth)
    viewport_mtx = repeat(viewport_mtx, 'b h w -> (repeat b) h w', repeat=batch_size)

    B = src_image.shape[0]
    H, W = src_image.shape[2:4]
    src_scr_mtx = (viewport_mtx @ src_proj_mtx).to(src_proj_mtx)
    mvp_mtx = (tar_proj_mtx @ rel_view_mtx).to(rel_view_mtx)

    # Coordinate grids.
    grid: Float[Tensor, 'H W C'] = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device, dtype=dtype)

    # Unproject depth.
    screen = F.pad(grid, (0, 1), 'constant', 0)  # z=0 (z doesn't matter)
    screen = F.pad(screen, (0, 1), 'constant', 1)  # w=1
    screen = repeat(screen, 'h w c -> b h w c', b=B)
    screen_flat = rearrange(screen, 'b h w c -> b (h w) c')
    # To eye coordinates.
    eye = screen_flat @ torch.linalg.inv_ex(src_scr_mtx.float())[0].mT.to(dtype)
    # Overwrite depth.
    eye = eye * rearrange(src_depth, 'b c h w -> b (h w) c')
    eye[..., 3] = 1

    # Coordinates embedding.
    coords = torch.stack((grid[..., 0] / H, grid[..., 1] / W), dim=-1)
    embedder = get_embedder(2)
    embed = repeat(embedder(coords), 'h w c -> b c h w', b=B)

    # Warping.
    input_image: Float[Tensor, 'B C H W'] = torch.cat([embed, src_image], dim=1)
    output_image = forward_warper(input_image, screen_flat[..., :2], eye, mvp_mtx=mvp_mtx, viewport_mtx=viewport_mtx)
    # warped_embed = output_image['warped'][:, : embed.shape[1]]
    warped_image = output_image['warped'][:, embed.shape[1] :]

    return warped_image


def warp_function_batch(
    src_image: Float[Tensor, 'B C H W'],
    src_depth: Float[Tensor, 'B 1 H W'],
    rel_view_mtx: Float[Tensor, 'B F 4 4'],  # (B, F, 4, 4)
    src_proj_mtx: Float[Tensor, 'B 4 4'],  # (B, 4, 4)
    tar_proj_mtx: Float[Tensor, 'B 4 4'],  # (B, 4, 4)
    # 其余必要参数可酌情加
) -> Float[Tensor, 'B F C H W']:
    """
    批量的Warp函数，一次性处理 (B, F) 个相对视角变换。
    最终输出形状为 (B, F, C, H, W)。
    """

    device = src_image.device
    dtype = src_image.dtype

    # ========== 1) 预处理 ==========
    # 这里沿用与 warp_function 相同的预处理思路
    src_image = preprocess_image(src_image)  # (B, C, 512, 512)
    src_depth = preprocess_image(src_depth)  # (B, 1, 512, 512)

    B, C, H, W = src_image.shape
    F_ = rel_view_mtx.shape[1]  # 视频帧数

    # ========== 2) viewport_mtx ==========
    # 在原 warp_function 中是 get_viewport_matrix(512, 512, batch_size=1,...)
    # 这里要改为 batch_size=B，再加上F维度
    viewport_mtx = get_viewport_matrix(width=512, height=512, batch_size=B, device=device).to(dtype)  # (B, 4, 4)
    # 扩展到 (B, F, 4, 4)
    viewport_mtx = viewport_mtx.unsqueeze(1).expand(-1, F_, -1, -1)  # (B, F, 4, 4)

    # ========== 3) 构建 src_scr_mtx = viewport_mtx @ src_proj_mtx ==========
    # 先把 src_proj_mtx 从 (B,4,4) 扩维到 (B,F,4,4)
    src_proj_mtx = src_proj_mtx.unsqueeze(1).expand(-1, F_, -1, -1)  # (B, F, 4, 4)
    src_scr_mtx = viewport_mtx @ src_proj_mtx  # (B, F, 4, 4)

    # ========== 4) 构建 mvp_mtx = tar_proj_mtx @ rel_view_mtx ==========
    tar_proj_mtx = tar_proj_mtx.unsqueeze(1).expand(-1, F_, -1, -1)  # (B, F, 4, 4)
    mvp_mtx = tar_proj_mtx @ rel_view_mtx  # (B, F, 4, 4)

    # ========== 5) 坐标网格与反投影 ==========
    # 原 warp_function 做法：先在 (H, W) 构建 grid，再 pad => z=0, w=1
    grid = torch.stack(torch.meshgrid(torch.arange(W, device=device, dtype=dtype), torch.arange(H, device=device, dtype=dtype), indexing='xy'), dim=-1)  # (W, H, 2)
    # 变成 (H, W, 2)
    grid = grid.permute(1, 0, 2).contiguous()  # (H, W, 2)

    # 在原函数中：  screen => (B, H, W, 4)
    screen = F.pad(grid, (0, 1), 'constant', 0)  # (H, W, 3), z=0
    screen = F.pad(screen, (0, 1), 'constant', 1)  # (H, W, 4), w=1

    # 此处需要在 batch(F) 维度上都同样的 screen，所以:
    # 先扩展到 (B, F, H, W, 4)
    screen = screen.unsqueeze(0).unsqueeze(1).expand(B, F_, H, W, 4)
    # 再展平成 (B, F, H*W, 4)
    screen_flat = rearrange(screen, 'b f h w c -> b f (h w) c')

    # ========== 6) 计算 src_scr_mtx 的逆并反投影 (unproject) ==========
    #   src_scr_mtx: (B, F, 4, 4)
    #   我们需要逆矩阵：inv_scr_mtx => (B, F, 4, 4)
    #   PyTorch 2.0+ 有 torch.linalg.inv_ex，可先 flatten 后再reshape
    BF = B * F_
    src_scr_mtx_bf = rearrange(src_scr_mtx, 'b f h w -> (b f) h w')  # (BF, 4,4)
    inv_scr_mtx_bf = torch.linalg.inv_ex(src_scr_mtx_bf.float())[0].to(dtype)
    inv_scr_mtx = inv_scr_mtx_bf.view(B, F_, 4, 4)  # (B, F, 4, 4)

    # 做矩阵乘法时，我们可以先把 screen_flat reshape => (BF, H*W, 4)
    screen_flat_bf = rearrange(screen_flat, 'b f n c -> (b f) n c')  # (BF, H*W, 4)
    # eye => (BF, H*W, 4)
    eye_bf = screen_flat_bf @ inv_scr_mtx_bf.mT  # (BF, H*W, 4)

    # ========== 7) 融合 depth (把 z 覆盖或相乘) ==========
    # src_depth: (B, 1, H, W) => 同样扩展到 (B, F, H, W)
    #   如果所有帧使用同样的深度(即相同相机), 也可以 repeat, 具体看需求
    src_depth_f = src_depth.expand(B, F_, -1, -1)  # (B, F, H, W)
    src_depth_f = src_depth_f.unsqueeze(2)
    src_depth_f_bf = rearrange(src_depth_f, 'b f c h w -> (b f) (c) (h w)', c=1)
    # => (BF, 1, H*W)
    # 进一步恢复 => (BF, H*W, 1)
    src_depth_f_bf = src_depth_f_bf.permute(0, 2, 1).contiguous()  # (BF, H*W, 1)

    # 替换 eye_bf[..., :1] => 只要 z
    eye_bf = eye_bf * src_depth_f_bf
    # w=1
    eye_bf[..., 3] = 1

    # ========== 8) 构建坐标嵌入 + 拼接到图像通道 ==========
    # 这里与原函数类似： embedder(coords)
    # coords 大小 (H, W, 2)，先 repeat 到 (B, F, H, W, 2)
    # 并把结果 permute 成 (B, F, C_embed, H, W) => flatten => (BF, C_embed, H, W)
    coords = torch.stack((grid[..., 0] / H, grid[..., 1] / W), dim=-1)  # (H, W, 2)
    embedder = get_embedder(2)
    embed_2d = embedder(coords)  # (H, W, C_embed)
    # 扩展到 (B, F, H, W, C_embed)
    embed_2d = embed_2d.unsqueeze(0).unsqueeze(0).expand(B, F_, H, W, -1)
    # 变成 (B, F, C_embed, H, W)
    embed_2d = embed_2d.permute(0, 1, 4, 2, 3).contiguous()
    # 变成 (BF, C_embed, H, W)
    embed_2d = rearrange(embed_2d, 'b f c h w -> (b f) c h w')

    # 源图像也需要扩展到F维度 => (B, F, C, H, W) => (BF, C, H, W)
    src_img_f = src_image.unsqueeze(1).expand(-1, F_, -1, -1, -1)  # (B, F, C, H, W)
    src_img_f_bf = rearrange(src_img_f, 'b f c h w -> (b f) c h w')

    # 拼接 => (BF, C_embed + C_src, H, W)
    input_image_bf = torch.cat([embed_2d, src_img_f_bf], dim=1)

    # ========== 9) 调用 forward_warper ==========
    # forward_warper 需要的参数:
    #   image: (BF, C, H, W)
    #   screen: (BF, H*W, 2)
    #   pcd: (BF, H*W, 4)
    #   mvp_mtx: (BF, 4, 4)
    #   viewport_mtx: (BF, 4, 4)
    #
    # 先 reshape mvp_mtx, viewport_mtx => (BF, 4,4)
    mvp_mtx_bf = rearrange(mvp_mtx, 'b f h w -> (b f) h w')
    viewport_mtx_bf = rearrange(viewport_mtx, 'b f h w -> (b f) h w')

    # screen_flat_bf: (BF, H*W, 4) => forward_warper 只需要 [:, :, :2]
    screen_2d_bf = screen_flat_bf[..., :2]  # (BF, H*W, 2)

    # pcd => eye_bf => (BF, H*W, 4)
    # 调用 forward_warper
    output = forward_warper(image=input_image_bf, screen=screen_2d_bf, pcd=eye_bf, mvp_mtx=mvp_mtx_bf, viewport_mtx=viewport_mtx_bf, alpha=0.5)  # 或你使用的其它超参
    warped_bf = output['warped']  # (BF, C_out, H, W)
    # 其中 C_out = embed大小 + src_image 通道数；最后几通道才是图像

    # ========== 10) 截取出 warped_image 部分 ==========
    # 假设前 embed_2d.shape[1] 通道是 embed，那 warped_bf[:, embed_dim:] 便是图像
    embed_dim = embed_2d.shape[1]
    warped_image_bf = warped_bf[:, embed_dim:]  # (BF, C, H, W)

    # ========== 11) reshape 回 (B, F, C, H, W) ==========
    warped_image = rearrange(warped_image_bf, '(b f) c h w -> b f c h w', b=B, f=F_)

    return warped_image


def get_mask_batch(
    first_frames: torch.Tensor,  # (B, H, W, 3)
    depths: torch.Tensor,  # (B, H, W)
    camera_poses: torch.Tensor,  # (B, F, 19)
    ori_hs: torch.Tensor,  # (B,)
    ori_ws: torch.Tensor,  # (B,)
    res: int,
):
    """
    将原先对 batch_size 的循环转换成批量处理。
    每个 batch 的第一帧图像 + 深度 + 一系列 camera_poses，
    要对所有帧 (F) 做 warp，并得到一个二值 mask。

    返回结果 mask 形状: (B, F, H, W)
    """
    device = first_frames.device
    dtype = first_frames.dtype

    B, H, W, _ = first_frames.shape
    _, F, _ = camera_poses.shape

    # --------------------------
    # 1. 相机内参与外参准备
    # --------------------------
    # 取出源相机外参
    #   camera_poses[..., 7:] => shape=(B, F, 12)，reshape -> (B, F, 3, 4)
    src_wc = camera_poses[:, 0, 7:].reshape(B, 3, 4)  # 第 0 帧作为 source

    # 取出焦距信息，用于 get_src_proj_mtx
    focal_length_x = camera_poses[:, 0, 1]  # (B,)
    focal_length_y = camera_poses[:, 0, 2]  # (B,)

    # 先把 first_frames, depths 转到 [B, C, H, W] 以适配 warp_function
    # first_frames: (B, 3, H, W), depths: (B, 1, H, W)
    first_frames_t = first_frames.permute(0, 3, 1, 2).contiguous()
    depths_t = depths.unsqueeze(1).contiguous()

    # --------------------------
    # 2. 构建投影矩阵
    # --------------------------
    # 分批算出投影矩阵 (B, 4, 4)
    src_proj_mtx = get_src_proj_mtx_batch(focal_length_x, focal_length_y, ori_hs, ori_ws, res, first_frames_t)
    tar_proj_mtx = src_proj_mtx  # 目前源与目标相同

    # --------------------------
    # 3. 为所有帧一次性计算相对外参，并做批量 Warp
    # --------------------------
    # tar_wc => (B, F, 3, 4)
    tar_wc = camera_poses[..., 7:].reshape(B, F, 3, 4)

    # 统一做相对变换: (B, F, 4, 4)
    rel_view_mtx = get_rel_view_mtx_batch(src_wc, tar_wc).to(device=device, dtype=dtype)

    # 进行批量 warp，得到 (B, F, 3, H, W) 的 warped 图像
    # 注意：需要令 warp_function 支持一个“额外的帧维度 F”
    # 这里示例写了 warp_function_batch，如有需要可自行封装
    warped = warp_function_batch(
        first_frames_t,
        depths_t,
        rel_view_mtx,  # (B, F, 4, 4)
        src_proj_mtx,  # (B, 4, 4)
        tar_proj_mtx,  # (B, 4, 4)
    )
    # warped => (B, F, 3, H, W)

    # --------------------------
    # 4. 将 warped 转成灰度并进行二值化得到 mask
    # --------------------------
    # 假设简单取通道均值作为灰度
    warped_gray = warped.mean(dim=2)  # -> (B, F, H, W)

    # 当像素 == 0 时，我们设定 mask = 1，否则为 0
    # 注意，这里阈值可根据实际需求调整
    eps = 1e-7
    mask = (warped_gray.abs() < eps).to(torch.uint8)  # (B, F, H, W)
    return mask, warped
