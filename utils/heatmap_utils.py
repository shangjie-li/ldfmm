import torch
from torch.nn import functional as F


def nms_heatmap(heatmap, kernel=3):
    """
    Suppress small values around maximum values.
    e.g.
        0.4 0.4 0.4      0.0 0.0 0.0
        0.4 0.7 0.4  ->  0.0 0.7 0.0
        0.4 0.4 0.4      0.0 0.0 0.0
    """
    padding = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=(kernel, kernel), stride=1, padding=padding)
    eq_index = (hmax == heatmap).float()

    return heatmap * eq_index


def select_topk(heatmap, K=100):
    """
    Select top K scores in the heatmap.

    Args:
        heatmap: tensor of float, [B, C, H, W]
        K: int, top k samples to be selected

    Returns:
        topk_scores_all: tensor of float, [B, K], top scores
        topk_inds: tensor of int, [B, K], indices (0 to H * W - 1)
        topk_classes: tensor of float, [B, K], classes (0 to num_classes - 1)
        topk_xs: tensor of float, [B, K]
        topk_ys: tensor of float, [B, K]

    """
    batch_size, num_classes, height, width = heatmap.shape

    heatmap = heatmap.view(batch_size, -1)  # [B, C * H * W]
    topk_scores_all, topk_inds_all = torch.topk(heatmap, K)  # [B, K], [B, K]

    topk_inds = topk_inds_all % (height * width)
    topk_xs = (topk_inds % width).float()
    topk_ys = (topk_inds / width).float()

    topk_classes = (topk_inds_all / (height * width)).float().floor()

    return topk_scores_all, topk_inds, topk_classes, topk_xs, topk_ys


def get_poi(features, indices):
    """
    Get POI (points of interest) features in the feature map by keypoints or indices.

    Args:
        features: tensor of float, [B, C, H, W], regression feature map
        indices: tensor of int, in point format [B, K, 2] or index format [B, K]

    Returns:
        features: tensor of float, [B, K, C], selected features

    """
    batch_size, num_channels, height, width = features.shape
    if len(indices.shape) == 3:
        indices = indices[:, :, 1] * width + indices[:, :, 0]  # [B, K]

    features = features.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
    features = features.view(batch_size, -1, num_channels)  # [B, H * W, C]
    indices = indices.unsqueeze(-1).repeat(1, 1, num_channels).long()  # [B, K, C]

    # Select specific features based on POIs.
    features = features.gather(dim=1, index=indices)  # [B, K, C]

    return features
