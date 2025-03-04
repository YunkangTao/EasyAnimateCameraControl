import os
from einops import rearrange
import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    PNDMScheduler,
)
from omegaconf import OmegaConf
from PIL import Image
from transformers import (
    BertModel,
    BertTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    Qwen2Tokenizer,
    Qwen2VLForConditionalGeneration,
    T5EncoderModel,
    T5Tokenizer,
)

from easyanimate.models import name_to_autoencoder_magvit, name_to_transformer3d
from easyanimate.models.transformer3d import get_teacache_coefficients
from easyanimate.pipeline.pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline
from easyanimate.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import get_video_to_video_latent, save_videos_grid

# from EasyCamera.warp_a_video import *
from scripts.train_inpainting_5_1_with_depth import prepare_depth_anything, pre_process_first_frames, get_inpaint_latents_from_depth
from EasyCamera.tools import save_videos_set
from safetensors.torch import load_file
from easyanimate.data.dataset_inpainting_with_depth import VideoDatasetWithDepth
import torch.nn.functional as F


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
    enable_teacache,
    teacache_threshold,
    config_path,
    model_name,
    sampler_name,
    transformer_path,
    depth_anything_path,
    lora_path,
    sample_size,
    video_length,
    fps,
    weight_dtype,
    denoise_strength,
    guidance_scale,
    seed,
    num_inference_steps,
    lora_weight,
    data_json,
    data_path,
    negative_prompt,
    save_path,
):

    config = OmegaConf.load(config_path)

    # Get the dataset
    original_fps = 25
    frame_skip = 1 if fps is None else int(original_fps // fps)
    train_dataset = VideoDatasetWithDepth(
        data_json,
        data_path,
        video_sample_size=sample_size,
        video_sample_stride=frame_skip,
        video_sample_n_frames=video_length,
        enable_bucket=False,
        enable_inpaint=True,
    )

    # Get Transformer
    Choosen_Transformer3DModel = name_to_transformer3d[config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel')]

    transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
    if weight_dtype == torch.float16 and "v5.1" not in model_name.lower():
        transformer_additional_kwargs["upcast_attention"] = True

    transformer = Choosen_Transformer3DModel.from_pretrained_2d(
        model_name,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
        torch_dtype=torch.float8_e4m3fn if GPU_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
        low_cpu_mem_usage=True,
    )

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if motion_module_path is not None:
        print(f"From Motion Module: {motion_module_path}")
        if motion_module_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(motion_module_path)
        else:
            state_dict = torch.load(motion_module_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}, {u}")

    # Get DepthAnythingV2
    depth_anything = prepare_depth_anything(config['depth_anything_kwargs']['dav2_model'], config['depth_anything_kwargs']['dav2_outdoor'])
    depth_anything.to("cuda", dtype=weight_dtype)

    # Get Vae
    Choosen_AutoencoderKL = name_to_autoencoder_magvit[config['vae_kwargs'].get('vae_type', 'AutoencoderKL')]
    vae = Choosen_AutoencoderKL.from_pretrained(model_name, subfolder="vae", vae_additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])).to(weight_dtype)
    if weight_dtype == torch.float16 and "v5.1" not in model_name.lower():
        vae.upcast_vae = True

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
        tokenizer = BertTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
            tokenizer_2 = Qwen2Tokenizer.from_pretrained(os.path.join(model_name, "tokenizer_2"))
        else:
            tokenizer_2 = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    else:
        if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
            tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(model_name, "tokenizer"))
        else:
            tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
        tokenizer_2 = None

    if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
        text_encoder = BertModel.from_pretrained(model_name, subfolder="text_encoder").to(weight_dtype)
        if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
            text_encoder_2 = Qwen2VLForConditionalGeneration.from_pretrained(
                os.path.join(model_name, "text_encoder_2"),
                torch_dtype=weight_dtype,
            )
        else:
            text_encoder_2 = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2").to(weight_dtype)
    else:
        if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
            text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                os.path.join(model_name, "text_encoder"),
                torch_dtype=weight_dtype,
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder").to(weight_dtype)
        text_encoder_2 = None

    if transformer.config.in_channels != vae.config.latent_channels and config['transformer_additional_kwargs'].get('enable_clip_in_inpaint', True):
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_name, subfolder="image_encoder").to("cuda", weight_dtype)
        clip_image_processor = CLIPImageProcessor.from_pretrained(model_name, subfolder="image_encoder")
    else:
        clip_image_encoder = None
        clip_image_processor = None

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM": DDIMScheduler,
        "Flow": FlowMatchEulerDiscreteScheduler,
    }[sampler_name]

    scheduler = Choosen_Scheduler.from_pretrained(model_name, subfolder="scheduler")

    pipeline = EasyAnimateInpaintPipeline(
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
        clip_image_processor=clip_image_processor,
    )

    if GPU_memory_mode == "sequential_cpu_offload":
        pipeline._manual_cpu_offload_in_sequential_cpu_offload = []
        for name, _text_encoder in zip(["text_encoder", "text_encoder_2"], [pipeline.text_encoder, pipeline.text_encoder_2]):
            if isinstance(_text_encoder, Qwen2VLForConditionalGeneration):
                if hasattr(_text_encoder, "visual"):
                    del _text_encoder.visual
                convert_model_weight_to_float8(_text_encoder)
                convert_weight_dtype_wrapper(_text_encoder, weight_dtype)
                pipeline._manual_cpu_offload_in_sequential_cpu_offload = [name]
        pipeline.enable_sequential_cpu_offload()
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        for _text_encoder in [pipeline.text_encoder, pipeline.text_encoder_2]:
            if hasattr(_text_encoder, "visual"):
                del _text_encoder.visual
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()

    coefficients = get_teacache_coefficients(model_name)
    if coefficients is not None and enable_teacache:
        print(f"Enable TeaCache with threshold: {teacache_threshold}.")
        pipeline.transformer.enable_teacache(num_inference_steps, teacache_threshold, coefficients=coefficients)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    if vae.cache_mag_vae:
        video_length = int((video_length - 1) // vae.mini_batch_encoder * vae.mini_batch_encoder) + 1 if video_length != 1 else 1
    else:
        video_length = int(video_length // vae.mini_batch_encoder * vae.mini_batch_encoder) if video_length != 1 else 1

    for index in range(len(train_dataset)):
        sample = train_dataset[index]

        pixel_values = sample["pixel_values"].unsqueeze(0).to("cuda")  # (B, F, 3, 512, 512)
        camera_poses = sample["camera_poses"].unsqueeze(0).to("cuda")
        text = sample["text"]
        type = sample["type"]
        idx = sample["idx"]
        ori_hs = torch.tensor(sample["ori_h"])
        ori_ws = torch.tensor(sample["ori_w"])
        video_id = sample["video_id"]
        title = sample["title"]
        first_frames = sample["clip_pixel_values"].unsqueeze(0).to("cuda")

        # input_video, input_video_mask, clip_image = get_video_to_video_latent(
        #     validation_video,
        #     video_length=video_length,
        #     fps=fps,
        #     validation_video_mask=validation_video_mask,
        #     sample_size=sample_size,
        # )

        first_frames_processed, (h, w) = pre_process_first_frames(first_frames, "cuda", weight_dtype)  # 2,3,518,518
        first_frames_processed = rearrange(first_frames_processed, "b f c h w -> (b f) c h w")
        with torch.no_grad():
            depths = depth_anything.forward(first_frames_processed)  # torch.Size([2, 518, 518])
        depths = F.interpolate(depths.unsqueeze(1), (h, w), mode="bilinear", align_corners=True).squeeze(1)  # torch.Size([2, 512, 512])
        depths = rearrange(depths, "(b f) h w -> b f h w", f=video_length)  # torch.Size([1, 49, 512, 512])

        latents_shape = (1, 16, 13, 64, 64)

        t2v_flag, mask_pixel_values_with_noise, mask, mask_reshape, mask_pixel_values, mask_warped = get_inpaint_latents_from_depth(
            depths,
            first_frames,
            camera_poses,
            ori_hs,
            ori_ws,
            sample_size[0],
            pixel_values,  # (B, F, 3, 512, 512)
            weight_dtype,
            "cuda",
            latents_shape,  # (B, 16, 13, 64, 64)
            vae.cache_mag_vae,
        )
        # pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
        mask_warped = rearrange(mask_warped, "b f c h w -> b c f h w")
        # input_video = pixel_values * 0.5 + 0.5
        input_video_mask = mask * 255
        input_video_mask = input_video_mask.unsqueeze(1)  # (B, 1, F, H, W)
        clip_image = None

        with torch.no_grad():
            sample = pipeline(
                prompt=text,
                video_length=video_length,
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                # video=input_video,
                mask_video=input_video_mask,
                masked_video_latents=mask_warped,
                clip_image=clip_image,
                strength=denoise_strength,
            ).frames
        sample = sample.permute([0, 2, 1, 3, 4])
        # pixel_values = rearrange(pixel_values, "b c f h w -> b f c h w")
        mask_warped = rearrange(mask_warped, "b c f h w -> b f c h w")

        video_path = os.path.join(save_path, video_id)
        dir_path = os.path.dirname(video_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        save_camera_path = video_path.replace(".mp4", ".txt")

        save_videos_set(first_frames, depths, mask, mask_warped, sample, pixel_values, video_path, fps)
        save_camera_pose(save_camera_path, camera_poses, title)

    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device="cuda", dtype=weight_dtype)


if __name__ == '__main__':

    # GPU memory mode, which can be choosen in [model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    #
    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use,
    # and the transformer model has been quantized to float8, which can save more GPU memory.
    #
    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use,
    # resulting in slower speeds but saving a large amount of GPU memory.
    #
    # EasyAnimateV3 support "model_cpu_offload" "sequential_cpu_offload"
    # EasyAnimateV4, V5 and V5.1 support "model_cpu_offload" "model_cpu_offload_and_qfloat8" "sequential_cpu_offload"
    GPU_memory_mode = "model_cpu_offload_and_qfloat8"
    # EasyAnimateV5.1 support TeaCache.
    enable_teacache = True
    # Recommended to be set between 0.05 and 0.1. A larger threshold can cache more steps, speeding up the inference process,
    # but it may cause slight differences between the generated content and the original content.
    teacache_threshold = 0.08

    # Config and model path
    config_path = "config/easyanimate_video_v5.1_magvit_qwen_with_depth.yaml"
    model_name = "models/Diffusion_Transformer/EasyAnimateV5.1-7b-zh-InP"

    # Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM" "Flow"
    # EasyAnimateV3 support "Euler" "Euler A" "DPM++" "PNDM"
    # EasyAnimateV4 and V5 support "Euler" "Euler A" "DPM++" "PNDM" "DDIM".
    # EasyAnimateV5.1 supports Flow.
    sampler_name = "Flow"

    # Load pretrained model if need
    transformer_path = None
    depth_anything_path = None
    # Only V1 does need a motion module
    motion_module_path = None
    vae_path = None
    lora_path = "output_dir_20250301_inpainting_with_depth_lora/checkpoint-4019.safetensors"

    # Other params
    sample_size = [512, 512]
    # In EasyAnimateV3, V4, the video_length of video is 1 ~ 144.
    # In EasyAnimateV5, V5.1, the video_length of video is 1 ~ 49.
    # If u want to generate a image, please set the video_length = 1.
    video_length = 49
    fps = 8

    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16
    # If you are preparing to redraw the reference video, set validation_video and validation_video_mask.
    # If you do not use validation_video_mask, the entire video will be redrawn;
    # if you use validation_video_mask, as shown in asset/mask.jpg, only a portion of the video will be redrawn.
    # Please set a larger denoise_strength when using validation_video_mask, such as 1.00 instead of 0.70
    denoise_strength = 1.0

    guidance_scale = 6.0
    seed = 43
    num_inference_steps = 50
    lora_weight = 0.55

    data_json = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/EvaluationSet/RealEstate10KBeforeProcess/metadata.json"
    data_path = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/EvaluationSet/RealEstate10KBeforeProcess"

    negative_prompt = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code, Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art, quiet, solid."

    save_path = "output_dir_20250301_inpainting_with_depth_lora/results_lora_checkpoint-4019"

    main(
        GPU_memory_mode,
        enable_teacache,
        teacache_threshold,
        config_path,
        model_name,
        sampler_name,
        transformer_path,
        depth_anything_path,
        lora_path,
        sample_size,
        video_length,
        fps,
        weight_dtype,
        denoise_strength,
        guidance_scale,
        seed,
        num_inference_steps,
        lora_weight,
        data_json,
        data_path,
        negative_prompt,
        save_path,
    )
