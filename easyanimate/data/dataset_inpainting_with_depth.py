import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from threading import Thread

# import albumentations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
from typing import Dict, List

VIDEO_READER_TIMEOUT = 20


# def get_random_mask(shape):
#     f, c, h, w = shape

#     if f != 1:
#         mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05])
#     else:
#         mask_index = np.random.choice([0, 1], p=[0.2, 0.8])
#     mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

#     if mask_index == 0:
#         center_x = torch.randint(0, w, (1,)).item()
#         center_y = torch.randint(0, h, (1,)).item()
#         block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
#         block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

#         start_x = max(center_x - block_size_x // 2, 0)
#         end_x = min(center_x + block_size_x // 2, w)
#         start_y = max(center_y - block_size_y // 2, 0)
#         end_y = min(center_y + block_size_y // 2, h)
#         mask[:, :, start_y:end_y, start_x:end_x] = 1
#     elif mask_index == 1:
#         mask[:, :, :, :] = 1
#     elif mask_index == 2:
#         mask_frame_index = np.random.randint(1, 5)
#         mask[mask_frame_index:, :, :, :] = 1
#     elif mask_index == 3:
#         mask_frame_index = np.random.randint(1, 5)
#         mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
#     elif mask_index == 4:
#         center_x = torch.randint(0, w, (1,)).item()
#         center_y = torch.randint(0, h, (1,)).item()
#         block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
#         block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

#         start_x = max(center_x - block_size_x // 2, 0)
#         end_x = min(center_x + block_size_x // 2, w)
#         start_y = max(center_y - block_size_y // 2, 0)
#         end_y = min(center_y + block_size_y // 2, h)

#         mask_frame_before = np.random.randint(0, f // 2)
#         mask_frame_after = np.random.randint(f // 2, f)
#         mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
#     elif mask_index == 5:
#         mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
#     elif mask_index == 6:
#         num_frames_to_mask = random.randint(1, max(f // 2, 1))
#         frames_to_mask = random.sample(range(f), num_frames_to_mask)

#         for i in frames_to_mask:
#             block_height = random.randint(1, h // 4)
#             block_width = random.randint(1, w // 4)
#             top_left_y = random.randint(0, h - block_height)
#             top_left_x = random.randint(0, w - block_width)
#             mask[i, 0, top_left_y : top_left_y + block_height, top_left_x : top_left_x + block_width] = 1
#     elif mask_index == 7:
#         center_x = torch.randint(0, w, (1,)).item()
#         center_y = torch.randint(0, h, (1,)).item()
#         a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
#         b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

#         for i in range(h):
#             for j in range(w):
#                 if ((i - center_y) ** 2) / (b**2) + ((j - center_x) ** 2) / (a**2) < 1:
#                     mask[:, :, i, j] = 1
#     elif mask_index == 8:
#         center_x = torch.randint(0, w, (1,)).item()
#         center_y = torch.randint(0, h, (1,)).item()
#         radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
#         for i in range(h):
#             for j in range(w):
#                 if (i - center_y) ** 2 + (j - center_x) ** 2 < radius**2:
#                     mask[:, :, i, j] = 1
#     elif mask_index == 9:
#         for idx in range(f):
#             if np.random.rand() > 0.5:
#                 mask[idx, :, :, :] = 1
#     else:
#         raise ValueError(f"The mask_index {mask_index} is not define")
#     return mask


class VideoSamplerWithDepth(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self, sampler: Sampler, dataset: Dataset, batch_size: int, drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, ' f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, ' f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket: Dict[str, List[int]] = {'GenXD': [], 'kubric': [], 'objaverse': [], 'realestate': [], 'VidGen': []}

    def __iter__(self):
        for idx in self.sampler:
            sample = self.dataset[idx]
            content_type = sample.get('type', 'GenXD')

            if content_type in self.bucket:
                self.bucket[content_type].append(idx)
                # 检查当前类型的桶是否达到 batch_size
                if len(self.bucket[content_type]) == self.batch_size:
                    yield self.bucket[content_type][:]
                    self.bucket[content_type].clear()
            else:
                raise ValueError(f"Unknown content type: {content_type}")

        # 处理剩余的索引
        if not self.drop_last:
            for content_type, indices in self.bucket.items():
                if len(indices) > 0:
                    yield indices[:]
                    self.bucket[content_type].clear()


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames


def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)

    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame


def process_mask(masks, h, w):
    """
    处理 mask，将其从 (25, 512, 512, 1) 转换为 (25, 1, 256, 256) 的 torch.uint8 张量，
    且值为 0 或 1。

    Args:
        masks (np.ndarray): 输入的 mask，形状为 (25, 512, 512, 1)，值范围 0-255
        h (int): 输出的高度，默认 256
        w (int): 输出的宽度，默认 256

    Returns:
        torch.Tensor: 处理后的 mask，形状为 (25, 1, 256, 256)，dtype=torch.uint8，值为 0 或 1
    """
    # 去除最后一个维度，形状变为 (25, 512, 512)
    masks = np.squeeze(masks, axis=-1)

    # 转换为浮点型的 torch 张量
    masks = torch.from_numpy(masks).float()  # (25, 512, 512)

    # 添加通道维度，形状变为 (25, 1, 512, 512)
    masks = masks.unsqueeze(1)

    # 使用双线性插值将 mask 缩放到 (256, 256)
    # 对于二值 mask，建议使用 'nearest' 模式以避免插值引入中间值
    masks = F.interpolate(masks, size=(h, w), mode='nearest')

    # 将 mask 的值二值化为 0 或 1
    masks = (masks > 128).to(torch.uint8)

    return masks


def prepare_camera_poses(camera_pose_file):
    whole_camera_para = []

    with open(camera_pose_file, 'r', encoding='utf-8') as file:
        # 读取所有行
        lines = file.readlines()

        title = lines[0].strip()

        # 确保文件至少有两行
        if len(lines) < 2:
            print("文件内容不足两行，无法读取数据。")
            return whole_camera_para

        # 跳过第一行，从第二行开始处理
        for idx, line in enumerate(lines[1:], start=2):
            # 去除首尾空白字符并按空格分割
            parts = line.strip().split()

            # 检查每行是否有19个数字
            if len(parts) != 19:
                print(f"警告：第 {idx} 行的数字数量不是19，跳过该行。")
                continue

            try:
                # 将字符串转换为浮点数
                numbers = [float(part) for part in parts]
                whole_camera_para.append(numbers)
            except ValueError as ve:
                print(f"警告：第 {idx} 行包含非数字字符，跳过该行。错误详情: {ve}")
                continue

    return title, whole_camera_para


class VideoDatasetWithDepth(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=512,
        video_sample_stride=4,
        video_sample_n_frames=49,
        # image_sample_size=512,
        # video_repeat=0,
        text_drop_ratio=-1,
        enable_bucket=False,
        video_length_drop_start=0.1,
        video_length_drop_end=0.9,
        enable_inpaint=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            self.dataset = json.load(open(ann_path))

        self.data_root = data_root

        # # It's used to balance num of images and videos.
        # self.dataset = []
        # for data in dataset:
        #     if data.get('type', 'image') != 'video':
        #         self.dataset.append(data)
        # if video_repeat > 0:
        #     for _ in range(video_repeat):
        #         for data in dataset:
        #             if data.get('type', 'image') == 'video':
        #                 self.dataset.append(data)
        # del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # # Image params
        # self.image_sample_size = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        # self.image_transforms = transforms.Compose(
        #     [
        #         transforms.Resize(min(self.image_sample_size)),
        #         transforms.CenterCrop(self.image_sample_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        #     ]
        # )

        # self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        video_id, camera_id, text, data_type = data_info['video_file_path'], data_info['camera_file_path'], data_info['text'], data_info['type']

        if self.data_root is None:
            video_dir = video_id
            camera_dir = camera_id
        else:
            video_dir = os.path.join(self.data_root, video_id)
            camera_dir = os.path.join(self.data_root, camera_id)

        title, camera_poses = prepare_camera_poses(camera_dir)

        with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
            min_sample_n_frames = min(self.video_sample_n_frames, int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride))
            if min_sample_n_frames == 0:
                raise ValueError(f"No Frames in video.")

            video_length = int(self.video_length_drop_end * len(video_reader))
            clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
            start_idx = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                sample_args = (video_reader, batch_index)
                pixel_values = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args)
                resized_frames = []
                for i in range(len(pixel_values)):
                    frame = pixel_values[i]
                    resized_frame = resize_frame(frame, self.video_sample_size)
                    resized_frames.append(resized_frame)

                pixel_values = np.array(resized_frames)
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.0
                del video_reader
            else:
                pixel_values = pixel_values

            if not self.enable_bucket:
                pixel_values = self.video_transforms(pixel_values)

            # Random use no text generation
            if random.random() < self.text_drop_ratio:
                text = ''

        camera_poses = [camera_poses[i] for i in batch_index]

        return pixel_values, camera_poses, text, data_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # data_info = self.dataset[idx % len(self.dataset)]
        # data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                # data_info_local = self.dataset[idx % len(self.dataset)]
                # data_type_local = data_info_local.get('type', 'image')
                # if data_type_local != data_type:
                #     raise ValueError("data_type_local != data_type")

                pixel_values, camera_poses, text, data_type = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["camera_poses"] = camera_poses
                sample["text"] = text
                sample["type"] = data_type
                sample["idx"] = idx

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length - 1)

        if self.enable_inpaint and not self.enable_bucket:
            # mask = get_random_mask(pixel_values.size())
            # mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            # sample["mask_pixel_values"] = mask_pixel_values
            # sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            # ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            # if (mask == 1).all():
            #     ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            # sample["ref_pixel_values"] = ref_pixel_values

        return sample


if __name__ == "__main__":
    # dataset = VideoDatasetWithDepth(ann_path="test.json")
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16)
    # for idx, batch in enumerate(dataloader):
    #     print(batch["pixel_values"].shape, len(batch["text"]))
    from bucket_sampler import RandomSampler

    train_dataset = VideoDatasetWithDepth(
        "datasets/z_mini_datasets_warped_videos_2_3_test/metadata.json",
        "datasets/z_mini_datasets_warped_videos_2_3_test",
        video_sample_size=512,
        video_sample_stride=3,
        video_sample_n_frames=49,
        enable_bucket=False,
        enable_inpaint=True,
    )

    # DataLoaders creation:
    batch_sampler_generator = torch.Generator().manual_seed(42)
    batch_sampler = VideoSamplerWithDepth(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, 1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        persistent_workers=True,
        num_workers=1,
    )

    for epoch in range(0, 100):
        batch_sampler.sampler.generator = torch.Generator().manual_seed(42 + epoch)
        for step, batch in enumerate(train_dataloader):
            print(
                f"Epoch: {epoch}, Step: {step}, idx: {batch['idx']}, pixel_values: {batch['pixel_values'].shape}, ground_truth: {batch['ground_truth'].shape}, type: {batch['type']}, mask_pixel_values: {batch['mask_pixel_values'].shape}, mask: {batch['mask'].shape}, clip_pixel_values: {batch['clip_pixel_values'].shape}, ref_pixel_values: {batch['ref_pixel_values'].shape}"
            )
