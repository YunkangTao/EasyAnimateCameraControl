import json
import os

import cv2
import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from omegaconf import OmegaConf
from PIL import Image
from transformers import (
    BertModel,
    BertTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5Tokenizer,
)

from easyanimate.models import name_to_autoencoder_magvit, name_to_transformer3d
from easyanimate.pipeline.pipeline_easycamera import EasyCameraPipeline
from easyanimate.utils.fp8_optimization import convert_weight_dtype_wrapper
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import get_video_to_video_latent, save_videos_grid

# from EasyCamera.warp_a_video import *
from scripts.train_inpainting_with_depth import prepare_depth_anything
from EasyCamera.model import EasyCamera
from EasyCamera.tools import save_videos_set
from safetensors.torch import load_file
from easyanimate.data.dataset_inpainting_with_depth import VideoDatasetWithDepth


# def get_video_to_video_latent_with_mask(input_video_path, video_length, sample_size):
#     if isinstance(input_video_path, str):
#         cap = cv2.VideoCapture(input_video_path)
#         input_video = []

#         frame_skip = 1
#         frame_count = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if frame_count % frame_skip == 0:
#                 # frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
#                 input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#             frame_count += 1

#             if frame_count >= video_length:
#                 break

#         cap.release()
#     else:
#         input_video = input_video_path

#     split_frames = [[] for _ in range(6)]
#     # 拆分每一帧并存储到对应的列表中

#     for idx, frame in enumerate(input_video):
#         height, width, channels = frame.shape
#         num_rows = 2
#         num_cols = 3
#         split_width = 512  # 每个子帧的宽度
#         split_height = 512  # 每个子帧的高度

#         expected_width = split_width * num_cols  # 3 * 512 = 1536
#         expected_height = split_height * num_rows  # 2 * 512 = 1024
#         if width != expected_width or height != expected_height:
#             print(f"第 {idx} 帧的尺寸 {width}x{height} 不符合预期 {expected_width}x{expected_height}，跳过此帧。")
#             continue

#         # 逐行逐列拆分帧
#         for row in range(num_rows):
#             for col in range(num_cols):
#                 index = row * num_cols + col  # 计算子帧的索引（0到5）
#                 start_y = row * split_height
#                 end_y = (row + 1) * split_height
#                 start_x = col * split_width
#                 end_x = (col + 1) * split_width

#                 # 提取子帧
#                 sub_frame = frame[start_y:end_y, start_x:end_x]

#                 # 将子帧添加到对应的列表中
#                 split_frames[index].append(sub_frame)

#     input_video = torch.from_numpy(np.array(split_frames[2]))
#     input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

#     input_video_mask = torch.from_numpy(np.array(split_frames[4]))
#     input_video_mask = input_video_mask.permute([3, 0, 1, 2]).unsqueeze(0)
#     input_video_mask = (input_video_mask > 128).all(dim=1, keepdim=True)
#     input_video_mask = input_video_mask * 255
#     input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)

#     # output_video = torch.from_numpy(np.array(split_frames[5]))
#     # output_video = output_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

#     # validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
#     # input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)

#     # input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
#     # input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
#     # input_video_mask = input_video_mask.to(input_video.device, input_video.dtype) # torch.Size([1, 1, 49, 384, 672])

#     return input_video, input_video_mask


# def get_video_to_video_latent_with_depth(
#     video_file,
#     camera_pose_file,
#     video_length,
#     sample_size,
#     depth_anything_path,
#     dav2_outdoor,
#     dav2_model,
# ):
#     depth_anything = prepare_models(dav2_outdoor, dav2_model, depth_anything_path)
#     frames, width, height = prepare_frames(video_file, video_length)
#     camera_poses = prepare_camera_poses(camera_pose_file, video_length)
#     res = sample_size[0]

#     output_frames = []
#     src_frame = frames[0]
#     src_camera_pose = camera_poses[0]

#     focal_length_x = src_camera_pose[1]
#     focal_length_y = src_camera_pose[2]
#     principal_point_x = src_camera_pose[3]
#     principal_point_y = src_camera_pose[4]

#     for frame, camera_pose in tqdm(zip(frames, camera_poses), total=len(frames), desc="Processing frames"):
#         with torch.no_grad():
#             warped_image = process_one_frame(
#                 src_frame,
#                 src_camera_pose,
#                 frame,
#                 camera_pose,
#                 width,
#                 height,
#                 focal_length_x,
#                 focal_length_y,
#                 principal_point_x,
#                 principal_point_y,
#                 res,
#                 depth_anything,
#             )
#         output_frames.append(warped_image)

#     warped_tensor = torch.stack(output_frames, dim=1).permute([0, 2, 1, 3, 4])  # torch.Size([1, 3, 49, 512, 512])
#     warped_gray = warped_tensor.mean(dim=1, keepdim=True)  # -> (B, 1, F, H, W)

#     eps = 1e-7
#     mask_tensor = (warped_gray.abs() < eps).to(torch.uint8)  # (B, 1, F, H, W)
#     mask_tensor = mask_tensor * 255  # torch.Size([1, 1, 49, 512, 512])

#     return warped_tensor, mask_tensor


def save_camera_pose(save_camera_path, camera_poses, title):
    """
    将相机参数（tensor格式）写回到指定的文件中。

    参数：
        save_camera_path (str): 文件路径
        title (str): 文件首行标题
        camera_poses (torch.Tensor): 形状为 [1, 49, 19] 的tensor，每一行包含19个数
    """
    # 将 tensor 转换为列表
    para_list = camera_poses.tolist()

    # 如果 tensor 的第一个维度为1（即只有一个batch），则去掉这个维度
    if isinstance(para_list, list) and len(para_list) == 1:
        para_list = para_list[0]

    with open(save_camera_path, 'w', encoding='utf-8') as file:
        # 写入标题行
        file.write(title.strip() + "\n")

        # 从第二行开始写入每一组相机参数
        for idx, params in enumerate(para_list, start=2):
            if len(params) != 19:
                print(f"警告：第 {idx} 行的参数数量不是19，将跳过该行。")
                continue
            # 将每个数字转换为字符串，以空格分隔后写入文件
            line = " ".join(str(num) for num in params)
            file.write(line + "\n")


def main(
    GPU_memory_mode,
    config_path,
    model_name,
    sampler_name,
    weight_dtype,
    guidance_scale,
    seed,
    num_inference_steps,
    checkpoint_path,
    sample_size,
    video_length,
    fps,
    denoise_strength,
    data_json,
    data_path,
    negative_prompt,
    save_path,
):

    config = OmegaConf.load(config_path)

    # Get the dataset
    train_dataset = VideoDatasetWithDepth(
        data_json,
        data_path,
        video_sample_size=sample_size,
        video_sample_stride=1,
        video_sample_n_frames=video_length,
        enable_bucket=False,
        enable_inpaint=True,
    )

    # Get Transformer
    Choosen_Transformer3DModel = name_to_transformer3d[config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel')]

    transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
    if weight_dtype == torch.float16:
        transformer_additional_kwargs["upcast_attention"] = True

    transformer = Choosen_Transformer3DModel.from_pretrained_2d(
        model_name,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
        torch_dtype=torch.float8_e4m3fn if GPU_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
        low_cpu_mem_usage=True,
    )

    # Get DepthAnythingV2
    depth_anything = prepare_depth_anything(config['depth_anything_kwargs']['dav2_model'], config['depth_anything_kwargs']['dav2_outdoor'])

    easycamera = EasyCamera(transformer, depth_anything)
    if not hasattr(easycamera, "dtype"):
        easycamera.dtype = weight_dtype

    if checkpoint_path is not None:
        print(f"From checkpoint: {checkpoint_path}")

        # 确保目录存在
        transformer_dir = os.path.join(checkpoint_path, "transformer")
        depth_dir = os.path.join(checkpoint_path, "dav2")

        if not os.path.exists(transformer_dir):
            raise FileNotFoundError(f"Transformer directory not found at {transformer_dir}")
        if not os.path.exists(depth_dir):
            raise FileNotFoundError(f"Dav2 directory not found at {depth_dir}")

        # load transformer
        transformer_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
        state_dict = load_file(transformer_path)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = easycamera.easyanimate.load_state_dict(state_dict, strict=False)
        print(f"load transformer: missing keys: {len(m)}, unexpected keys: {len(u)}")

        # load depth anything
        depth_path = os.path.join(depth_dir, "depth_anything.pth")
        easycamera.depth_anything_v2.load_state_dict(torch.load(depth_path))

    # Get Vae
    Choosen_AutoencoderKL = name_to_autoencoder_magvit[config['vae_kwargs'].get('vae_type', 'AutoencoderKL')]
    vae = Choosen_AutoencoderKL.from_pretrained(model_name, subfolder="vae", vae_additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])).to(weight_dtype)
    if config['vae_kwargs'].get('vae_type', 'AutoencoderKL') == 'AutoencoderKLMagvit' and weight_dtype == torch.float16:
        vae.upcast_vae = True

    if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
        tokenizer = BertTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        tokenizer_2 = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
        tokenizer_2 = None

    if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
        text_encoder = BertModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=weight_dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=weight_dtype)
    else:
        text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=weight_dtype)
        text_encoder_2 = None

    if transformer.config.in_channels != vae.config.latent_channels and config['transformer_additional_kwargs'].get('enable_clip_in_inpaint', True):
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_name, subfolder="image_encoder").to("cuda", weight_dtype)
        clip_image_processor = CLIPImageProcessor.from_pretrained(model_name, subfolder="image_encoder")
    else:
        clip_image_encoder = None
        clip_image_processor = None

    # Get Scheduler
    Choosen_Scheduler = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM": DDIMScheduler,
    }[sampler_name]
    scheduler = Choosen_Scheduler.from_pretrained(model_name, subfolder="scheduler")

    pipeline = EasyCameraPipeline.from_pretrained(
        model_name,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        vae=vae,
        easycamera=easycamera,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        clip_image_encoder=clip_image_encoder,
        clip_image_processor=clip_image_processor,
    )

    if GPU_memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload()
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        pipeline.enable_model_cpu_offload()
        convert_weight_dtype_wrapper(pipeline.easycamera, weight_dtype)
    else:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    if vae.cache_mag_vae:
        video_length = int((video_length - 1) // vae.mini_batch_encoder * vae.mini_batch_encoder) + 1 if video_length != 1 else 1
    else:
        video_length = int(video_length // vae.mini_batch_encoder * vae.mini_batch_encoder) if video_length != 1 else 1

    for index in range(len(train_dataset)):
        sample = train_dataset[index]

        pixel_values = sample["pixel_values"].unsqueeze(0).to("cuda")
        camera_poses = sample["camera_poses"].unsqueeze(0).to("cuda")
        text = sample["text"]
        type = sample["type"]
        idx = sample["idx"]
        ori_h = sample["ori_h"]
        ori_w = sample["ori_w"]
        video_id = sample["video_id"]
        title = sample["title"]
        clip_pixel_values = sample["clip_pixel_values"].unsqueeze(0).to("cuda")

        with torch.no_grad():
            video, depths, mask, mask_warped = pipeline(
                text,
                video_length=video_length,
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                video=pixel_values,
                clip_image=clip_pixel_values,
                strength=denoise_strength,
                camera_poses=camera_poses,
                ori_h=ori_h,
                ori_w=ori_w,
                type=type,
                return_dict=False,
            )
        video = video.permute([0, 2, 1, 3, 4])

        video_path = os.path.join(save_path, video_id)
        dir_path = os.path.dirname(video_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        save_camera_path = video_path.replace(".mp4", ".txt")

        # # index = len([path for path in os.listdir(save_path)]) + 1
        # # prefix = str(index).zfill(8)

        # if video_length == 1:
        #     # save_sample_path = os.path.join(save_path, prefix + f".png")

        #     image = sample[0, :, 0]
        #     image = image.transpose(0, 1).transpose(1, 2)
        #     image = (image * 255).numpy().astype(np.uint8)
        #     image = Image.fromarray(image)
        #     image.save(save_path)
        # else:
        #     # video_path = os.path.join(save_path, prefix + ".mp4")
        #     # save_videos_grid(sample, video_path, fps=fps)

        save_videos_set(clip_pixel_values, depths, mask, mask_warped, video, pixel_values, video_path, fps)
        save_camera_pose(save_camera_path, camera_poses, title)


if __name__ == '__main__':

    # GPU memory mode, which can be choosen in [model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    #
    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use,
    # and the transformer model has been quantized to float8, which can save more GPU memory.
    #
    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use,
    # resulting in slower speeds but saving a large amount of GPU memory.
    GPU_memory_mode = "model_cpu_offload"

    # Config and model path
    config_path = "config/easyanimate_video_v5_magvit_multi_text_encoder_with_depth.yaml"
    model_name = "models/Diffusion_Transformer/EasyAnimateV5-7b-zh-InP"

    # Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
    # EasyAnimateV1, V2 and V3 cannot use DDIM.
    # EasyAnimateV4 and V5 support DDIM.
    sampler_name = "DDIM"

    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16
    # If you want to generate from text, please set the validation_image_start = None and validation_image_end = None

    guidance_scale = 6.0
    seed = 43
    num_inference_steps = 50
    lora_weight = 0.55

    # Load pretrained model if need
    checkpoint_path = "output_dir_20250219_inpainting_with_depth_transformer/checkpoint-8038"
    # checkpoint_path = None

    # Other params
    sample_size = [512, 512]
    video_length = 49
    fps = 14

    denoise_strength = 1.0

    data_json = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/EvaluationSet/RealEstate10KBeforeProcess/metadata.json"
    data_path = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/EvaluationSet/RealEstate10KBeforeProcess"

    negative_prompt = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code, Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art, quiet, solid."

    save_path = "output_dir_20250219_inpainting_with_depth_transformer/results_checkpoint-8038"

    main(
        GPU_memory_mode,
        config_path,
        model_name,
        sampler_name,
        weight_dtype,
        guidance_scale,
        seed,
        num_inference_steps,
        checkpoint_path,
        sample_size,
        video_length,
        fps,
        denoise_strength,
        data_json,
        data_path,
        negative_prompt,
        save_path,
    )
