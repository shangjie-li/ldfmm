import torch


def focal_loss(inputs, targets, gamma=2.0):
    """
    For a ground truth heatmap, like
        0.0 0.6 0.8 0.4 0.0
        0.3 0.7 1.0 0.7 0.1
        0.0 0.4 0.8 0.2 0.0
    the positive and negative mask are
        0.0 0.0 0.0 0.0 0.0    1.0 1.0 1.0 1.0 1.0
        0.0 0.0 1.0 0.0 0.0    1.0 1.0 0.0 1.0 1.0
        0.0 0.0 0.0 0.0 0.0    1.0 1.0 1.0 1.0 1.0
    the focal loss makes the network learn to predict the keypoint.

    Args:
        inputs: tensor of float32, [B, C, H, W], predicted heatmap (from 0 to 1)
        targets: tensor of float32, [B, C, H, W], ground truth heatmap (from 0 to 1)
        gamma: float

    Returns:
        loss: tensor of float, []

    """
    pos_mask = targets.eq(1).float()
    neg_mask = targets.lt(1).float()

    pos_loss = pos_mask * torch.log(inputs) * torch.pow(1 - inputs, gamma)
    neg_loss = neg_mask * torch.log(1 - inputs) * torch.pow(inputs, gamma) * torch.pow(1 - targets, 4)

    pos_loss = -pos_loss.sum()
    neg_loss = -neg_loss.sum()

    loss = pos_loss + neg_loss
    num_pos = pos_mask.float().sum()
    loss /= num_pos if num_pos >= 1.0 else 1.0

    return loss
