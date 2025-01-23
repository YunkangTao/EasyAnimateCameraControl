import torch
from PIL import Image
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as pl
import os
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images, load_images_from_pil_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import json
from tqdm import tqdm
import torch.nn.functional as F


def get_match_points_from_dust3r(pixel_values, mask_pixel_values, K=1000):
    """
    Args:
        pixel_values: ground truth, torch.Size([batch, 49, 3, 512, 512])
        mask_pixel_values: masked pixel value is -1, others are ground truth pixel values,
        ##mask: masked value is 1, others are 0
        K: how many matched key points return

    Returns:
        warped_points: torch.size(b, frames, K, 2)
        gt_points: torch.size(b, frames, K, 2)
        zero padding
    """

    device = 'cuda'
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    warped_points, gt_points = imgs2match(mask_pixel_values, pixel_values, model, K, device)
    return warped_points, gt_points


def imgs2match(warped_frames, gt_frames, model, K, device='cuda', niter=300, lr=0.01, schedule='cosine', batch_size=1):
    b, f, _, _, _ = warped_frames.size()
    warped_points_batch = []
    gt_points_batch = []
    for m in range(b):
        warped_points = []
        gt_points = []
        for n in range(f):
            # images = load_images([os.path.join(out_path, '{}/{}_gt/frame_{:04d}.jpg'.format(a, b, frame_idx)), os.path.join(out_path, '{}/{}_warped/frame_{:04d}.jpg'.format(a, b, frame_idx))], size=512)
            images = load_images_from_pil_images([tensor2pil_image(warped_frames[m][n]), tensor2pil_image(gt_frames[m][n])], size=512)
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, model, device, batch_size=batch_size)
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']
            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
            loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

            # retrieve useful values from scene:
            imgs = scene.imgs
            focals = scene.get_focals()
            poses = scene.get_im_poses()
            pts3d = scene.get_pts3d()
            confidence_masks = scene.get_masks()

            # find 2D-2D matches between the two images
            from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

            pts2d_list, pts3d_list = [], []
            for i in range(2):
                conf_i = confidence_masks[i].cpu().numpy()
                pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
                pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
            try:
                reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
                # print(f'found {num_matches} matches')
            except Exception as e:
                # print("cannnot find matched points")
                break
            matches_im1 = pts2d_list[1][reciprocal_in_P2]
            matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
            if num_matches == 0:
                break
            n_viz = K
            if num_matches >= n_viz:
                match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                # match_idx_to_viz = np.round(np.linspace(0, n_viz, n_viz)).astype(int)
                viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
                warped_points.append(torch.tensor(viz_matches_im0))
                gt_points.append(torch.tensor(viz_matches_im1))
            else:
                warped_points.append(F.pad(torch.tensor(matches_im0), (0, 0, 0, n_viz - num_matches)))
                gt_points.append(F.pad(torch.tensor(matches_im1), (0, 0, 0, n_viz - num_matches)))

        stack_warped_points = torch.stack(warped_points)
        stack_gt_points = torch.stack(gt_points)
        if len(stack_warped_points) == 49:
            warped_points_batch.append(stack_warped_points)
            gt_points_batch.append(stack_gt_points)
        else:
            warped_points_batch.append(F.pad(stack_warped_points, (0, 0, 0, 0, 0, 49 - len(stack_warped_points))))
            gt_points_batch.append(F.pad(stack_gt_points, (0, 0, 0, 0, 0, 49 - len(stack_gt_points))))
    res_warped_points_batch = torch.stack(warped_points_batch)
    res_gt_points_batch = torch.stack(gt_points_batch)
    return res_warped_points_batch, res_gt_points_batch


if __name__ == '__main__':

    def get_tensor_from_video(video_path):
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sub_width = frame_width // 3
        sub_height = frame_height // 2

        video_frames1 = torch.zeros((frame_count, sub_height, sub_width, 3), dtype=torch.uint8)
        video_frames2 = torch.zeros((frame_count, sub_height, sub_width, 3), dtype=torch.uint8)

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames1[i] = torch.from_numpy(frame_rgb[0:sub_height, 2 * sub_width : 3 * sub_width])
            video_frames2[i] = torch.from_numpy(frame_rgb[sub_height : 2 * sub_height, 2 * sub_width : 3 * sub_width])
        cap.release()

        # video_frames = video_frames.permute(3, 0, 1, 2)
        return video_frames1.permute(0, 3, 1, 2).unsqueeze(0), video_frames2.permute(0, 3, 1, 2).unsqueeze(0)

    def tensor2pil_image(tensor):
        if tensor.max() <= 1.0:
            tensor = tensor.mul(255).byte()
        tensor = tensor.permute(1, 2, 0)
        numpy_array = tensor.numpy()
        pil_image = Image.fromarray(numpy_array, mode='RGB')
        # pil_image.show()
        return pil_image

    video_path = '/home/chenyang_lei/video_diffusion_models/dust3r/test/7J2mMggrR9Y/52ece28e102eedc4.mp4'
    warped_frames, gt_frames = get_tensor_from_video(video_path)

    warped_points, gt_points = get_match_points_from_dust3r(gt_frames, warped_frames, K=1000)
    print(warped_points.shape, gt_points.shape)
