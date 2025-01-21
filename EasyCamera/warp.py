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


def get_projection_matrix(fovy: Float[Tensor, 'B'], aspect_wh: float, near: float, far: float) -> Float[Tensor, 'B 4 4']:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(fovy / 2.0)  # add a negative sign here as the y axis is flipped in nvdiffrast output
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


def convert_camera_extrinsics(w2c):
    # 获取设备和数据类型，以确保缩放矩阵与w2c在同一设备和数据类型
    device = w2c.device
    dtype = w2c.dtype

    # 定义缩放矩阵，x和y轴取反，z轴保持不变
    S = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=dtype))

    # 将缩放矩阵应用于旋转和平移部分
    R = w2c[:, :3]  # 3x3
    t = w2c[:, 3]  # 3

    new_R = S @ R  # 矩阵乘法
    new_t = S @ t  # 向量乘法

    # 构建新的外参矩阵
    new_w2c = torch.cat((new_R, new_t.unsqueeze(1)), dim=1)  # 3x4

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


def get_viewport_matrix(
    width: int,
    height: int,
    batch_size: int = 1,
    device: torch.device = None,
) -> Float[Tensor, 'B 4 4']:
    N = torch.tensor([[width / 2, 0, 0, width / 2], [0, height / 2, 0, height / 2], [0, 0, 1 / 2, 1 / 2], [0, 0, 0, 1]], dtype=torch.float32, device=device)[None].repeat(
        batch_size, 1, 1
    )
    return N


def preprocess_image(image: Float[Tensor, 'B C H W']) -> Float[Tensor, 'B C H W']:
    image = F.interpolate(image, (512, 512))
    return image


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
    points_ndc = points_c / points_c[..., 3:4]
    # To screen.
    coords_new = points_ndc @ viewport_mtx.mT

    # Masking invalid pixels.
    invalid = coords_new[..., 2] <= 0
    coords_new[invalid] = -1000000 if coords_new.dtype == torch.float32 else -1e4

    # Calculate flow and importance for splatting.
    new_z = points_c[..., 2:3]
    flow = coords_new[..., :2] - screen[..., :2]
    ## Importance.
    importance = alpha / new_z
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
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


def get_mask_and_mask_pixel_value(numpy_clip_pixel_value, depth, camera_pose, ori_h, ori_w, res):
    src_camera_pose = camera_pose[0]
    src_wc = torch.tensor(src_camera_pose[7:]).reshape((3, 4))
    focal_length_x = src_camera_pose[1]
    focal_length_y = src_camera_pose[2]
    principal_point_x = src_camera_pose[3]
    principal_point_y = src_camera_pose[4]

    # Projection matrix.
    src_proj_mtx = get_src_proj_mtx(focal_length_x, focal_length_y, ori_h, ori_w, res, numpy_clip_pixel_value)
    ## Use the same projection matrix for the source and the target.
    tar_proj_mtx = src_proj_mtx

    mask = []

    for pose in camera_pose:
        tar_wc = torch.tensor(pose[7:]).reshape((3, 4))
        rel_view_mtx = get_rel_view_mtx(src_wc, tar_wc, numpy_clip_pixel_value)
        warped_image = warp_function(
            numpy_clip_pixel_value,
            depth,
            rel_view_mtx,
            src_proj_mtx,
            tar_proj_mtx,
            # viewport_mtx,
        )
        warped_pil = to_pil_image(warped_image[0])
        warped_array = np.array(warped_pil.convert('L'))
        mask_array = np.where(warped_array == 0, 1, 0).astype(np.uint8)

        mask_tensor = torch.tensor(mask_array, dtype=torch.uint8)
        mask.append(mask_tensor)

    mask = torch.stack(mask)

    return mask
