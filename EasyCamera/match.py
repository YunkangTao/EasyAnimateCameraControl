def get_match_points_from_dust3r(pixel_values, mask_pixel_values, mask):
    """
    Args:
        pixel_values: ground truth, torch.Size([batch, 49, 3, 512, 512])
        mask_pixel_values: masked pixel value is -1, others are ground truth pixel values,
        mask: masked value is 1, others are 0

    Returns:
        warped_points: torch.size(b, frames, K, 2)
        gt_points: torch.size(b, frames, K, 2)
        zero padding
    """
    return warped_points, gt_points
